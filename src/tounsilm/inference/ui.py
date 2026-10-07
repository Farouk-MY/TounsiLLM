"""
Web UI template for TounsiLM Interactive Playground.
Provides rich, responsive modern styling with native Right-to-Left (RTL) Arabic
rendering, Arabizi/Derja support, dynamic direction detection, and real-time generation.
"""

HTML_PAGE = """<!DOCTYPE html>
<html lang="tn" dir="auto">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TounsiLM Playground | تونسي ال ام</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #090d16;
            --bg-card: rgba(18, 24, 38, 0.85);
            --bg-card-hover: rgba(26, 35, 54, 0.95);
            --bg-input: #0f172a;
            --border-subtle: rgba(255, 255, 255, 0.08);
            --border-focus: #e70013;
            --primary: #e70013;
            --primary-hover: #c40010;
            --primary-glow: rgba(231, 0, 19, 0.25);
            --gold: #f59e0b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --radius-lg: 16px;
            --radius-md: 12px;
            --radius-sm: 8px;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', 'Cairo', -apple-system, sans-serif;
            background-color: var(--bg-base);
            background-image: 
                radial-gradient(at 0% 0%, rgba(231, 0, 19, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(245, 158, 11, 0.08) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            overflow-x: hidden;
        }

        header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1rem 2rem;
            background: rgba(9, 13, 22, 0.8);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--border-subtle);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .flag-badge {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            background: #e70013;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            box-shadow: 0 0 15px var(--primary-glow);
        }

        .brand-title {
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .brand-title span {
            background: linear-gradient(135deg, #fff 40%, var(--gold) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .badge-arch {
            background: rgba(231, 0, 19, 0.15);
            border: 1px solid rgba(231, 0, 19, 0.3);
            color: #ff6b77;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 3px 8px;
            border-radius: 20px;
        }

        .device-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85rem;
            color: var(--text-muted);
            background: rgba(255, 255, 255, 0.04);
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid var(--border-subtle);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            box-shadow: 0 0 8px #10b981;
        }

        main {
            flex: 1;
            max-width: 1200px;
            width: 100%;
            margin: 0 auto;
            padding: 2rem 1.5rem;
            display: grid;
            grid-template-columns: 1fr 340px;
            gap: 1.5rem;
        }

        @media (max-width: 880px) {
            main {
                grid-template-columns: 1fr;
            }
        }

        .chat-section {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .panel-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-lg);
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }

        .chips-container {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            margin-bottom: 1rem;
        }

        .chip {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-family: 'Cairo', sans-serif;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .chip:hover {
            background: rgba(231, 0, 19, 0.15);
            color: var(--text-main);
            border-color: rgba(231, 0, 19, 0.4);
            transform: translateY(-1px);
        }

        .input-box-wrapper {
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .prompt-input {
            width: 100%;
            min-height: 120px;
            background: var(--bg-input);
            border: 1.5px solid var(--border-subtle);
            border-radius: var(--radius-md);
            padding: 1rem;
            color: var(--text-main);
            font-size: 1.05rem;
            font-family: 'Cairo', 'Plus Jakarta Sans', sans-serif;
            line-height: 1.6;
            resize: vertical;
            outline: none;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .prompt-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-glow);
        }

        .input-actions {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .hint {
            font-size: 0.8rem;
            color: var(--text-muted);
        }

        .btn-generate {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: linear-gradient(135deg, var(--primary) 0%, #b80010 100%);
            color: white;
            border: none;
            padding: 10px 24px;
            font-size: 0.95rem;
            font-weight: 700;
            border-radius: var(--radius-md);
            cursor: pointer;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px var(--primary-glow);
        }

        .btn-generate:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(231, 0, 19, 0.4);
        }

        .btn-generate:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }

        .output-card {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            min-height: 200px;
        }

        .output-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .output-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .output-body {
            font-family: 'Cairo', sans-serif;
            font-size: 1.25rem;
            line-height: 1.8;
            color: var(--text-main);
            white-space: pre-wrap;
            direction: auto;
            text-align: right;
            unicode-bidi: plaintext;
        }

        .output-body.empty {
            color: var(--text-muted);
            font-style: italic;
            font-size: 0.95rem;
            text-align: center;
            margin: auto;
        }

        .prompt-highlight {
            color: #60a5fa;
            font-weight: 600;
        }

        .gen-highlight {
            color: #f8fafc;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .meta-stats {
            display: flex;
            gap: 16px;
            font-size: 0.8rem;
            color: var(--text-muted);
            padding-top: 0.75rem;
            border-top: 1px solid var(--border-subtle);
        }

        /* Sidebar Controls */
        .controls-section {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }

        .param-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            margin-bottom: 1rem;
        }

        .param-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .param-val {
            color: var(--gold);
            font-family: monospace;
        }

        .param-slider {
            -webkit-appearance: none;
            width: 100%;
            height: 6px;
            border-radius: 3px;
            background: #1e293b;
            outline: none;
        }

        .param-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: var(--primary);
            cursor: pointer;
            box-shadow: 0 0 8px var(--primary-glow);
        }

        .btn-copy {
            background: transparent;
            border: 1px solid var(--border-subtle);
            color: var(--text-muted);
            padding: 4px 10px;
            border-radius: var(--radius-sm);
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn-copy:hover {
            color: var(--text-main);
            border-color: rgba(255, 255, 255, 0.2);
        }
    </style>
</head>
<body>
    <header>
        <div class="brand">
            <div class="flag-badge">🇹🇳</div>
            <div>
                <div class="brand-title">
                    <span>TounsiLM</span>
                    <span class="badge-arch">300M Model</span>
                </div>
            </div>
        </div>
        <div class="device-status">
            <div class="status-dot"></div>
            <span id="device-label">Loading device...</span>
        </div>
    </header>

    <main>
        <section class="chat-section">
            <div class="panel-card">
                <div class="chips-container">
                    <button class="chip" onclick="setPrompt('عسلامة، شنوة أحوالك لاباس؟')">👋 عسلامة، شنوة أحوالك؟</button>
                    <button class="chip" onclick="setPrompt('شنوة أحسن بلاصة تحوس فيها في تونس في الصيف؟')">🏖️ أحسن بلاصة في تونس</button>
                    <button class="chip" onclick="setPrompt('3asslema, chnowa a7welek lyoum?')">💬 3asslema (Arabizi)</button>
                    <button class="chip" onclick="setPrompt('kifech nchri billet mta3 train men tounes l sousse?')">🚆 billet train (Derja)</button>
                </div>
                <div class="input-box-wrapper">
                    <textarea 
                        id="prompt-input" 
                        class="prompt-input" 
                        placeholder="أكتب مساج بالتونسي أو بالعرابيزي... (مثال: عسلامة شنوة أحوالك؟)" 
                        dir="auto"
                        oninput="detectDir(this)"
                    ></textarea>
                    <div class="input-actions">
                        <span class="hint">Press <strong>Ctrl + Enter</strong> to generate</span>
                        <button id="btn-submit" class="btn-generate" onclick="generateText()">
                            <span>توليد النص | Generate</span>
                            <span id="btn-spinner" style="display:none;">⏳</span>
                        </button>
                    </div>
                </div>
            </div>

            <div class="panel-card output-card">
                <div class="output-header">
                    <div class="output-title">النتيجة الناتجة | Generated Text</div>
                    <button class="btn-copy" onclick="copyResult()">📋 Copy</button>
                </div>
                <div id="output-content" class="output-body empty">
                    اكتب سؤالك أو مساجك الفوق واضغط على توليد النص...
                </div>
                <div id="output-stats" class="meta-stats" style="display:none;">
                    <span id="stat-tokens">Tokens: 0</span>
                    <span id="stat-latency">Latency: 0ms</span>
                </div>
            </div>
        </section>

        <aside class="controls-section">
            <div class="panel-card">
                <h3 style="margin-bottom: 1.25rem; font-size: 1rem; color: var(--text-main);">Generation Settings</h3>
                
                <div class="param-group">
                    <div class="param-header">
                        <span>Max New Tokens</span>
                        <span id="val-max-tokens" class="param-val">60</span>
                    </div>
                    <input type="range" class="param-slider" id="param-max-tokens" min="10" max="300" step="5" value="60" oninput="updateVal('max-tokens', this.value)">
                </div>

                <div class="param-group">
                    <div class="param-header">
                        <span>Temperature</span>
                        <span id="val-temp" class="param-val">0.75</span>
                    </div>
                    <input type="range" class="param-slider" id="param-temp" min="0.1" max="1.5" step="0.05" value="0.75" oninput="updateVal('temp', this.value)">
                </div>

                <div class="param-group">
                    <div class="param-header">
                        <span>Top-p (Nucleus)</span>
                        <span id="val-top-p" class="param-val">0.90</span>
                    </div>
                    <input type="range" class="param-slider" id="param-top-p" min="0.5" max="1.0" step="0.05" value="0.90" oninput="updateVal('top-p', this.value)">
                </div>

                <div class="param-group">
                    <div class="param-header">
                        <span>Repetition Penalty</span>
                        <span id="val-rep" class="param-val">1.15</span>
                    </div>
                    <input type="range" class="param-slider" id="param-rep" min="1.0" max="2.0" step="0.05" value="1.15" oninput="updateVal('rep', this.value)">
                </div>
            </div>

            <div class="panel-card">
                <h4 style="font-size: 0.9rem; margin-bottom: 0.75rem; color: var(--gold);">ℹ️ Arabic Script in Browsers</h4>
                <p style="font-size: 0.8rem; color: var(--text-muted); line-height: 1.6;">
                    Unlike the Windows PowerShell terminal, modern web browsers natively support Unicode HarfBuzz cursive shaping and bidirectional Right-to-Left (RTL) Arabic letters.
                </p>
            </div>
        </aside>
    </main>

    <script>
        function updateVal(id, val) {
            document.getElementById('val-' + id).innerText = val;
        }

        function setPrompt(text) {
            const input = document.getElementById('prompt-input');
            input.value = text;
            detectDir(input);
            input.focus();
        }

        function detectDir(el) {
            const text = el.value.trim();
            if (!text) return;
            const isArabic = /[\\u0600-\\u06FF]/.test(text.charAt(0));
            el.dir = isArabic ? 'rtl' : 'ltr';
        }

        document.getElementById('prompt-input').addEventListener('keydown', function(e) {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                generateText();
            }
        });

        async function fetchInfo() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                document.getElementById('device-label').innerText = data.device === 'cuda' ? 'NVIDIA GPU (CUDA) Online' : 'Running on CPU';
            } catch (err) {
                document.getElementById('device-label').innerText = 'Connected';
            }
        }
        fetchInfo();

        async function generateText() {
            const input = document.getElementById('prompt-input');
            const prompt = input.value.trim();
            if (!prompt) return;

            const btn = document.getElementById('btn-submit');
            const spinner = document.getElementById('btn-spinner');
            const outputBox = document.getElementById('output-content');
            const statsBox = document.getElementById('output-stats');

            btn.disabled = true;
            spinner.style.display = 'inline';

            outputBox.className = 'output-body';
            const isArabic = /[\\u0600-\\u06FF]/.test(prompt);
            outputBox.style.direction = isArabic ? 'rtl' : 'ltr';
            outputBox.style.textAlign = isArabic ? 'right' : 'left';
            outputBox.innerHTML = `<span class="prompt-highlight">${escapeHtml(prompt)}</span> <span style="opacity:0.5;">...جاري التوليد</span>`;

            const maxTokens = parseInt(document.getElementById('param-max-tokens').value);
            const temperature = parseFloat(document.getElementById('param-temp').value);
            const topP = parseFloat(document.getElementById('param-top-p').value);
            const repetitionPenalty = parseFloat(document.getElementById('param-rep').value);

            const startTime = performance.now();

            try {
                const res = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        max_new_tokens: maxTokens,
                        temperature: temperature,
                        top_p: topP,
                        repetition_penalty: repetitionPenalty,
                        top_k: 40
                    })
                });

                const data = await res.json();
                const elapsed = Math.round(performance.now() - startTime);

                if (res.ok) {
                    const fullText = data.generated_text;
                    const continuation = fullText.startsWith(prompt) ? fullText.slice(prompt.length) : fullText;
                    
                    outputBox.innerHTML = `<span class="prompt-highlight">${escapeHtml(prompt)}</span><span class="gen-highlight">${escapeHtml(continuation)}</span>`;
                    
                    statsBox.style.display = 'flex';
                    document.getElementById('stat-tokens').innerText = `Tokens: ${data.num_generated_tokens}`;
                    document.getElementById('stat-latency').innerText = `Latency: ${elapsed}ms`;
                } else {
                    outputBox.innerHTML = `<span style="color:#ef4444;">Error: ${data.detail || 'Generation failed'}</span>`;
                }
            } catch (err) {
                outputBox.innerHTML = `<span style="color:#ef4444;">Connection error: ${err.message}</span>`;
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
            }
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.innerText = text;
            return div.innerHTML;
        }

        function copyResult() {
            const outputBox = document.getElementById('output-content');
            if (outputBox.classList.contains('empty')) return;
            navigator.clipboard.writeText(outputBox.innerText);
            alert('Copied to clipboard!');
        }
    </script>
</body>
</html>
"""
