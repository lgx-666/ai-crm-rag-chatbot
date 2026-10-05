// static/login.js
document.getElementById('loginBtn').addEventListener('click', async () => {
    const phone = document.getElementById('phoneInput').value.trim();
    const password = document.getElementById('passwordInput').value.trim();
    const errorMsg = document.getElementById('errorMsg');

    if (!phone || !password) {
        errorMsg.textContent = '请填写手机号和密码';
        return;
    }

    try {
        const resp = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ phone_number: phone, password })
        });
        const data = await resp.json();
        if (resp.ok) {
            localStorage.setItem('token', data.token);
            localStorage.setItem('user', JSON.stringify(data.user));  // data.user 已是 safe_user，无密码
            window.location.href = '/chat_page';
        } else {
            errorMsg.textContent = data.error || '登录失败';
        }
    } catch (e) {
        errorMsg.textContent = '网络错误';
    }
});