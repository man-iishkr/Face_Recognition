let interval;

document.getElementById('startBtn').onclick = async () => {
  const fileInput = document.getElementById('excelFile');
  const file = fileInput.files[0];
  if (!file) { alert("Please select an Excel file."); return; }

  const formData = new FormData();
  formData.append('file', file);

  const uploadRes = await fetch('/upload', { method: 'POST', body: formData });
  const uploadData = await uploadRes.json();
  if (uploadData.status !== 'success') { alert(uploadData.message); return; }

  const startRes = await fetch('/start', { method: 'POST' });
  const startData = await startRes.json();
  if (startData.status !== 'started') { alert("Failed to start recognition."); return; }

  document.getElementById("statusText").textContent = "Recognition started...";
  document.getElementById("startBtn").disabled = true;
  document.getElementById("stopBtn").disabled = false;

  interval = setInterval(async () => {
    const res = await fetch('/status');
    const data = await res.json();
    document.getElementById("attendanceList").textContent = data.marked.join(", ") || "None";
    if (!data.marked.length && !startData.status) clearInterval(interval);
  }, 1000);
};
function updateDateTime() {
  const now = new Date();
  const dateOptions = { year: 'numeric', month: 'long', day: 'numeric' };
  const timeOptions = { hour: '2-digit', minute: '2-digit', second: '2-digit' };

  document.getElementById('current-date').textContent =
    '📅 Date: ' + now.toLocaleDateString(undefined, dateOptions);
  document.getElementById('current-time').textContent =
    '⏰ Time: ' + now.toLocaleTimeString(undefined, timeOptions);
}

// Start the time updater
setInterval(updateDateTime, 1000);
updateDateTime();

document.getElementById('stopBtn').onclick = async () => {
  await fetch('/stop', { method: 'POST' });
  document.getElementById("statusText").textContent = "Recognition stopped.";
  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;
  clearInterval(interval);
};
