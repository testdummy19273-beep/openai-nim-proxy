// server.js - OpenAI to NVIDIA NIM API Proxy
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));;

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// 🔥 REASONING DISPLAY TOGGLE - Shows/hides reasoning in output
const SHOW_REASONING = true; // Set to true to show reasoning with <think> tags

// 🔥 THINKING MODE TOGGLE - Enables thinking for specific models that support it
const ENABLE_THINKING_MODE = true; // Set to true to enable chat_template_kwargs thinking parameter

// Model mapping (adjust based on available NIM models)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'gpt-4': 'qwen/qwen3-coder-480b-a35b-instruct',
  'gpt-4-turbo': 'moonshotai/kimi-k2-instruct-0905',
  'gpt-4o': 'deepseek-ai/deepseek-v3.1',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'gemini-pro': 'qwen/qwen3-next-80b-a3b-thinking',
  'qwen-3.5-397b': 'qwen/qwen3.5-397b-a17b',
  'glm-4.7': 'z-ai/glm4.7',
  'glm-5': 'z-ai/glm5',
  'deepseek-3.2': 'deepseek-ai/deepseek-v3.2',
  'kimi-k2.5': 'moonshotai/kimi-k2.5',
  'nemotron-3-super': 'nvidia/nemotron-3-super-120b-a12b',
  'gemma-4-31b': 'google/gemma-4-31b-it',
  'deepseek-v4-pro': 'deepseek-ai/deepseek-v4-pro',
  'deepseek-v4-flash': 'deepseek-ai/deepseek-v4-flash'
};

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'ok', 
    service: 'OpenAI to NVIDIA NIM Proxy', 
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint (OpenAI compatible)
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(model => ({
    id: model,
    object: 'model',
    created: Date.now(),
    owned_by: 'nvidia-nim-proxy'
  }));
  
  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint (main proxy)
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;
    
    // Smart model selection with fallback
    let nimModel = MODEL_MAPPING[model];
    if (!nimModel) {
      try {
        await axios.post(`${NIM_API_BASE}/chat/completions`, {
          model: model,
          messages: [{ role: 'user', content: 'test' }],
          max_tokens: 1
        }, {
          headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
          validateStatus: (status) => status < 500
        }).then(res => {
          if (res.status >= 200 && res.status < 300) {
            nimModel = model;
          }
        });
      } catch (e) {}
      
      if (!nimModel) {
        const modelLower = model.toLowerCase();
        if (modelLower.includes('gpt-4') || modelLower.includes('claude-opus') || modelLower.includes('405b')) {
          nimModel = 'meta/llama-3.1-405b-instruct';
        } else if (modelLower.includes('claude') || modelLower.includes('gemini') || modelLower.includes('70b')) {
          nimModel = 'meta/llama-3.1-70b-instruct';
        } else {
          nimModel = 'meta/llama-3.1-8b-instruct';
        }
      }
    }
    
    // Transform OpenAI request to NIM format
    const nimRequest = {
      model: nimModel,
      messages: messages,
      temperature: temperature || 0.6,
      max_tokens: max_tokens || 9024,
      extra_body: ENABLE_THINKING_MODE ? { chat_template_kwargs: { thinking: true } } : undefined,
      stream: stream || false
    };
    
    // Make request to NVIDIA NIM API
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });
    
    if (stream) {
      // Handle streaming response with reasoning
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders(); 
      
      let buffer = '';
      let reasoningStarted = false;
      
      response.data.on('data', (chunk) => {
        buffer += chunk.toString('utf8');
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          const trimmedLine = line.trim();
          if (!trimmedLine) continue; 
          
          if (trimmedLine.startsWith('data: ')) {
            if (trimmedLine === 'data: [DONE]') {
              res.write('data: [DONE]\n\n');
              return;
            }
            
            try {
              const data = JSON.parse(trimmedLine.slice(6));
              const choice = data.choices?.[0];
              
              if (choice?.delta) {
                const delta = choice.delta;
                const reasoning = delta.reasoning_content;
                const content = delta.content;
                
                if (SHOW_REASONING) {
                  let combinedContent = '';
                  let hasUpdates = false;
                  
                  // 1. Process Reasoning
                  // By checking `if (reasoning)`, we ignore empty strings "" and nulls
                  if (reasoning) { 
                    hasUpdates = true;
                    if (!reasoningStarted) {
                      combinedContent += '<think>\n' + reasoning;
                      reasoningStarted = true;
                    } else {
                      combinedContent += reasoning;
                    }
                  }
                  
                  // 2. Process Standard Content
                  // By checking `if (content)`, we ensure empty strings "" don't accidentally close the tag
                  if (content) { 
                    hasUpdates = true;
                    if (reasoningStarted) {
                      combinedContent += '\n</think>\n\n' + content;
                      reasoningStarted = false;
                    } else {
                      combinedContent += content;
                    }
                  }
                  
                  // 3. Handle stream completion if model finishes while still "thinking"
                  if (choice.finish_reason && reasoningStarted) {
                    combinedContent += '\n</think>';
                    reasoningStarted = false;
                    hasUpdates = true;
                  }
                  
                  // 4. Apply updates
                  if (hasUpdates) {
                    delta.content = combinedContent;
                  } else if (content === '') {
                    // Only pass empty initialization chunks if we aren't mid-reasoning
                    if (!reasoningStarted) delta.content = '';
                  }
                  
                  delete delta.reasoning_content;
                  
                } else {
                  // Stripping reasoning
                  if (content !== undefined) {
                    delta.content = content || '';
                  } else {
                    delta.content = '';
                  }
                  delete delta.reasoning_content;
                }
              }
              
              res.write(`data: ${JSON.stringify(data)}\n\n`); 
            } catch (e) {
              res.write(`${trimmedLine}\n\n`);
            }
          }
        }
      });
      
      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });
    } else {
      // Handle NON-STREAMING response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';
          
          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
          }
          
          // Clean up the object to match OpenAI standard exactly
          const message = {
            role: choice.message.role,
            content: fullContent
          };
          
          return {
            index: choice.index,
            message: message,
            finish_reason: choice.finish_reason
          };
        }),
        usage: response.data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
      };
      
      res.json(openaiResponse);
    }
    
  } catch (error) {
    console.error('Proxy error:', error.message);
    
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
