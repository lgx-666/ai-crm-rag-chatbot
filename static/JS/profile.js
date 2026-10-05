// static/profile.js
const user = JSON.parse(localStorage.getItem('user') || '{}');
if (!user.phone_number) {
    window.location.href = '/static/login.html';
}

document.getElementById('userDisplay').textContent = `手机号: ${user.phone_number}`;
document.getElementById('nameInput').value = user.name || '';
document.getElementById('ageInput').value = user.age || '';
document.getElementById('occupationInput').value = user.occupation || '';
document.getElementById('interestInput').value = user.interest || '';

document.getElementById('saveBtn').addEventListener('click', async () => {
    const payload = {
        phone_number: user.phone_number,
        name: document.getElementById('nameInput').value,
        age: parseInt(document.getElementById('ageInput').value) || null,
        occupation: document.getElementById('occupationInput').value,
        interest: document.getElementById('interestInput').value
    };
    const resp = await fetch('/update_profile', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + localStorage.getItem('token')
        },
        body: JSON.stringify(payload)
    });
    const data = await resp.json();
    const msgDiv = document.getElementById('saveMsg');
    if (resp.ok) {
        localStorage.setItem('user', JSON.stringify(data.user));
        msgDiv.innerHTML = '<span style="color:#34c759;">✅ 保存成功</span>';
        document.getElementById('userDisplay').textContent = `手机号: ${data.user.phone_number}`;
    } else {
        msgDiv.innerHTML = `<span style="color:#ff3b30;">❌ ${data.error}</span>`;
    }
});

document.getElementById('logoutBtn').addEventListener('click', () => {
    localStorage.removeItem('user');
    localStorage.removeItem('phone_number');
    window.location.href = '/';    // ✅ 跳转到登录页
});