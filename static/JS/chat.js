// static/JS/chat.js
(function () {
    // 登录状态检查
    const token = localStorage.getItem('token');
    if (!token) { window.location.href = '/'; return; }   // 改成检查 token
    const user = JSON.parse(localStorage.getItem('user') || '{}');
    document.getElementById('userDisplay').textContent = user.phone_number || '未登录';

    // 退出登录
    document.getElementById('logoutBtn').addEventListener('click', () => {
        localStorage.removeItem('token')
        localStorage.removeItem('user');
        localStorage.removeItem('phone_number');
        window.location.href = '/';
    });

    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendButton = document.getElementById('sendButton');
    const typingIndicator = document.getElementById('typingIndicator');
    const errorMessage = document.getElementById('errorMessage');

    let isWaiting = false;

    // 创建消息元素（供加载历史和发送消息使用）
    function createMessageElement(role, content) {
        const wrapper = document.createElement('div');
        wrapper.className = `message ${role}`;
        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.textContent = role === 'user' ? '我' : 'AI';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = content;

        if (role === 'assistant') {
            wrapper.appendChild(avatar);
            wrapper.appendChild(bubble);
        } else {
            wrapper.appendChild(bubble);
            wrapper.appendChild(avatar);
        }
        return wrapper;
    }

    // 添加消息到界面（不保存到数据库，仅显示）
    function addMessage(role, content) {
        const el = createMessageElement(role, content);
        chatMessages.appendChild(el);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return el;
    }

    // ---------- 加载历史记录 ----------
    async function loadHistory() {
        try {
            const resp = await fetch('/chat_history', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + localStorage.getItem('token')
                }
            });
            const data = await resp.json();
            if (resp.ok && data.history) {
                // 清空聊天区域（清除初始欢迎消息）
                chatMessages.innerHTML = '';
                if (data.history.length === 0) {
                    // 没有历史，显示默认欢迎消息
                    const welcomeDiv = document.createElement('div');
                    welcomeDiv.className = 'message assistant';
                    welcomeDiv.innerHTML = `<div class="avatar">AI</div><div class="bubble">您好！我是 AI 知识助手，可以为您解答大模型和 Java 后端开发的相关问题。</div>`;
                    chatMessages.appendChild(welcomeDiv);
                } else {
                    // 逐条渲染历史消息
                    data.history.forEach(msg => {
                        const el = createMessageElement(msg.role, msg.content);
                        chatMessages.appendChild(el);
                    });
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }
            } else {
                console.warn('加载历史记录失败:', data.error);
            }
        } catch (e) {
            console.warn('加载历史记录网络错误:', e);
        }
    }

    // 显示错误
    function showError(msg) {
        errorMessage.textContent = msg;
        errorMessage.classList.add('show');
        setTimeout(() => errorMessage.classList.remove('show'), 5000);
    }

    // 发送消息
    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query || isWaiting) return;

        userInput.value = '';
        isWaiting = true;
        sendButton.disabled = true;
        typingIndicator.classList.add('active');

        // 添加用户消息（立即显示）
        addMessage('user', query);

        // 创建助手消息占位
        const assistantMsg = createMessageElement('assistant', '');
        chatMessages.appendChild(assistantMsg);
        const bubble = assistantMsg.querySelector('.bubble');
        chatMessages.scrollTop = chatMessages.scrollHeight;

        let fullContent = '';

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + localStorage.getItem('token')
                },
                body: JSON.stringify({ query: query })
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`服务器错误 (${response.status}): ${errText}`);
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });

                const events = buffer.split('\n\n');   // 按 SSE 事件分隔符切
                buffer = events.pop() || '';           // 最后一段可能没收全，留到下一轮拼

                for (const evt of events) {
                    const line = evt.trim();
                    if (!line.startsWith('data:')) continue;
                    const payload = line.slice(5).trim();       // 去掉 "data:" 前缀
                    if (payload === '[DONE]') continue;         // 结束标志，跳过
                    try {
                        const json = JSON.parse(payload);
                        if (json.content !== undefined) {
                            fullContent += json.content;
                            bubble.textContent = fullContent;
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        } else if (json.error) {
                            showError(json.error);
                        }
                    } catch (e) {
                        console.warn('解析SSE失败:', payload, e);
                    }
                }
            }
        } catch (error) {
            console.error('请求失败:', error);
            showError('网络错误或服务器异常，请稍后重试。');
            if (assistantMsg.parentNode) {
                assistantMsg.remove();
            }
        } finally {
            isWaiting = false;
            sendButton.disabled = false;
            typingIndicator.classList.remove('active');
            if (fullContent === '' && assistantMsg.parentNode) {
                assistantMsg.remove();
            }
        }
    }

    // 事件绑定
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });

    // ---------- 初始化：加载历史记录 ----------
    loadHistory();

    userInput.focus();
})();