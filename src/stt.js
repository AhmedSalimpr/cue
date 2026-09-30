// Speech-to-text factory. Decoupled from the LLM provider because Anthropic has
// no audio API — we transcribe with whatever audio-capable key is available, and
// fall back across providers. Returns { text, provider } or { text:'', error }.
const { pcmToWav } = require('./wav');
const { formatProviderErrorMessage, isQuotaError, CURRENT_GEMINI_DEFAULT } = require('./llm');

const BASE_VOCAB = 'CI/CD, Docker, Kubernetes, Terraform, Jenkins, AWS, Azure, GCP, ' +
  'CodeCommit, CodePipeline, CodeBuild, CodeDeploy, DevOps, SRE, microservices, deployment, ' +
  'pipeline, container, orchestration, Ansible, Prometheus, Grafana, Helm, EKS, ECS, Lambda, ' +
  'S3, EC2, IAM, GitHub Actions, GitLab, Kafka, PostgreSQL, Redis, MongoDB, REST API, gRPC';

// English-biasing prompt prepended to Groq/Whisper calls to anchor the model
const ENGLISH_BIAS_PREFIX = 'This is an English business meeting conversation. ';

function looksLikeHallucination(raw) {
  const trimmed = (raw || '').trim();
  if (!trimmed) return true;
  if (/^[\p{Emoji_Presentation}\p{Extended_Pictographic}\s]+$/u.test(trimmed)) return true;

  const t = trimmed.replace(/[.,!?…]+$/g, '').trim().toLowerCase();
  const clean = t.replace(/[^a-z\s]/g, '').replace(/\s+/g, ' ').trim();
  if (!clean) return true;

  // Detect word repetition loops (e.g. "erem erem erem", "you you you", "Meme, Meme.")
  const words = clean.split(/\s+/);
  if (words.length >= 2) {
    if (words.length === 2 && words[0] === words[1]) return true;
    if (words.length > 2) {
      const counts = {};
      let maxFreq = 0;
      for (const w of words) {
        counts[w] = (counts[w] || 0) + 1;
        if (counts[w] > maxFreq) maxFreq = counts[w];
      }
      if (maxFreq / words.length > 0.4) return true;
    }
  }

  // Detect foreign language hallucinations (non-ASCII characters e.g. Sjöndag, Díu, Híru)
  if (/[^\x00-\x7F]/.test(trimmed)) return true;

  // Detect list of capitalized proper nouns / names hallucinated from noise (e.g. "Enya, Hrabi, Bail", "Ayde, Held, Alveig")
  const rawWords = trimmed.split(/[\s,]+/);
  if (rawWords.length >= 2 && rawWords.every(w => /^[A-Z][a-z]+$/.test(w)) && /,/.test(trimmed)) {
    return true;
  }

  // Short transcript heuristic: ≤2 words that look like gibberish
  // Real English words rarely have 3+ consecutive consonants at start or unusual bigrams
  if (words.length <= 2) {
    const suspicious = words.filter(w =>
      /^[bcdfghjklmnpqrstvwxyz]{3,}/i.test(w) ||  // triple consonant start
      w.length <= 1 ||                              // single letter
      /(.)\1{2,}/.test(w) ||                        // triple repeated char
      (/^[qxzj]/i.test(w) && w.length <= 3)         // rare-start short word
    );
    if (suspicious.length === words.length) return true;
  }

  // Detect gibberish via letter entropy — real English words have common bigrams
  // If most words have unusual letter patterns, it's likely noise
  if (words.length >= 2 && words.length <= 6) {
    const uncommonWords = words.filter(w => {
      if (w.length < 3) return false;
      // Check for common English bigrams
      const commonBigrams = /th|he|in|er|an|re|on|at|en|nd|ti|es|or|te|of|ed|is|it|al|ar|st|to|nt|ng|se|ha|as|ou|io|le|ve|co|me|de|hi|ri|ro|ic|ne|ea|ra|ce/;
      const hasBigram = commonBigrams.test(w);
      return !hasBigram;
    });
    if (uncommonWords.length / words.length > 0.6) return true;
  }

  const artifacts = new Set([
    'thank you', 'thank you very much', 'thank you for watching', 'thanks for watching',
    'please subscribe', 'like and subscribe', 'bye-bye', 'bye bye', 'bye', 'you', 'okay',
    'kiss kill girls', 'oh fuck', 'you you', 'im sorry'
  ]);
  if (artifacts.has(t) || artifacts.has(clean)) return true;
  if (/^(you\s*)+$/.test(clean)) return true;
  if (/^(the\s*)+$/.test(clean)) return true;
  if (/^(thank you\s*)+$/.test(clean)) return true;

  return false;
}

function buildVocabPrompt(settings) {
  const s = settings || {};
  const text = (s.resumeText || '') + ' ' + (s.jobDescription || '');
  const proper = Array.from(new Set(text.match(/\b([A-Z][a-zA-Z0-9+.#]{2,}|[A-Z]{2,6})\b/g) || []));
  let prompt = BASE_VOCAB + (proper.length ? ', ' + proper.slice(0, 60).join(', ') : '');
  if (prompt.length > 850) prompt = prompt.slice(0, 850);
  return prompt;
}

async function transcribeOpenAI(apiKey, wav, model, baseURL, prompt) {
  const OpenAI = require('openai');
  const toFile = OpenAI.toFile || require('openai/uploads').toFile;
  const client = new OpenAI({ apiKey, baseURL });
  const file = await toFile(wav, 'audio.wav', { type: 'audio/wav' });
  const res = await client.audio.transcriptions.create({
    file,
    model: model || 'whisper-1',
    language: 'en',
    temperature: 0,
    prompt: prompt || ''
  });
  return (res.text || '').trim();
}

async function transcribeGemini(apiKey, wav) {
  const { GoogleGenAI } = require('@google/genai');
  const ai = new GoogleGenAI({ apiKey });
  const res = await ai.models.generateContent({
    model: CURRENT_GEMINI_DEFAULT,
    contents: [{ role: 'user', parts: [
      { text: 'Transcribe this audio verbatim. Return only the spoken words with no commentary. If there is no clear speech, return an empty response.' },
      { inlineData: { mimeType: 'audio/wav', data: wav.toString('base64') } }
    ] }]
  });
  return ((res && res.text) || '').trim();
}

function createSTT(settings) {
  const keys = settings.apiKeys || {};
  const selectedProvider = settings.sttProvider || 'auto';
  const vocabPrompt = buildVocabPrompt(settings);
  const groqPrompt = ENGLISH_BIAS_PREFIX + vocabPrompt;
  const chain = [];
  if ((selectedProvider === 'auto' || selectedProvider === 'openai') && keys.openai) {
    chain.push({ p: 'openai', fn: (wav) => transcribeOpenAI(keys.openai, wav, settings.sttModel, undefined, vocabPrompt) });
  }
  if ((selectedProvider === 'auto' || selectedProvider === 'groq') && keys.groq) {
    chain.push({ p: 'groq', fn: (wav) => transcribeOpenAI(keys.groq, wav, 'whisper-large-v3-turbo', 'https://api.groq.com/openai/v1', groqPrompt) });
  }
  if ((selectedProvider === 'auto' || selectedProvider === 'gemini') && keys.gemini) {
    chain.push({ p: 'gemini', fn: (wav) => transcribeGemini(keys.gemini, wav) });
  }
  if (keys.openai && chain.length > 1) chain.unshift(chain.splice(chain.findIndex((c) => c.p === 'openai'), 1)[0]);

  let disabledUntil = 0;
  let lastProvider = null;

  return {
    available: chain.length > 0,
    providers: chain.map((c) => c.p),
    async transcribe(pcm) {
      if (!chain.length || !pcm || pcm.length < 3200) return { text: '' };
      const now = Date.now();
      if (disabledUntil && now < disabledUntil) return { text: '', error: { provider: lastProvider, message: `Temporary ${lastProvider || 'provider'} quota or rate-limit; waiting 30s before retrying.` } };
      const wav = pcmToWav(pcm, 16000, 1);
      let lastErr = null;
      for (const c of chain) {
        try {
          const text = await c.fn(wav);
          disabledUntil = 0;
          lastProvider = c.p;
          if (looksLikeHallucination(text)) return { text: '', provider: c.p };
          return { text, provider: c.p };
        } catch (e) {
          // Shares detection/wording with the LLM error path (src/llm.js) so a
          // 404 (dead/misspelled model) or 429 (quota) reads the same whether it
          // came from a chat request or a transcription request.
          const quota = isQuotaError(e);
          const message = formatProviderErrorMessage(e, c.p);
          lastErr = { status: e && e.status, code: e && e.code, message, provider: c.p };
          if (quota) {
            lastProvider = c.p;
            disabledUntil = now + 30000;
            break;
          }
        }
      }
      return { text: '', error: lastErr };
    }
  };
}

module.exports = { createSTT, looksLikeHallucination, buildVocabPrompt };
