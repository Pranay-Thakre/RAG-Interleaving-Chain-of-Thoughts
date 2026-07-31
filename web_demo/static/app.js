document.addEventListener('DOMContentLoaded', () => {
  const queryForm = document.getElementById('queryForm');
  const datasetSelect = document.getElementById('datasetSelect');
  const systemSelect = document.getElementById('systemSelect');
  const retrievalSelect = document.getElementById('retrievalSelect');
  const questionInput = document.getElementById('questionInput');
  const submitBtn = document.getElementById('submitBtn');
  const samplePicker = document.getElementById('samplePicker');
  const stepTraceContainer = document.getElementById('stepTraceContainer');
  const paragraphContainer = document.getElementById('paragraphContainer');
  const executionBadge = document.getElementById('executionBadge');
  const paraCountBadge = document.getElementById('paraCountBadge');

  let datasetsData = [];

  // Fetch Datasets and Sample Questions
  fetch('/api/datasets')
    .then(res => res.json())
    .then(data => {
      datasetsData = data.datasets || [];
    })
    .catch(err => console.error('Failed to load datasets:', err));

  // Sample Question Click Handler
  samplePicker.addEventListener('click', () => {
    const selectedDsId = datasetSelect.value;
    const dsObj = datasetsData.find(d => d.id === selectedDsId);
    if (dsObj && dsObj.samples && dsObj.samples.length > 0) {
      const randomSample = dsObj.samples[Math.floor(Math.random() * dsObj.samples.length)];
      questionInput.value = randomSample;
    } else {
      questionInput.value = "Were Scott Derrickson and Ed Wood of the same nationality?";
    }
  });

  // Submit Handler
  queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();
    if (!question) return;

    // UI Loading state
    submitBtn.disabled = true;
    submitBtn.querySelector('.btn-text').textContent = 'Reasoning...';
    executionBadge.textContent = 'Executing IRCoT...';
    executionBadge.style.color = '#f59e0b';
    
    stepTraceContainer.innerHTML = '<div class="placeholder-msg"><div class="placeholder-icon">⏳</div><p>Interleaving Reasoning & Document Retrieval...</p></div>';
    paragraphContainer.innerHTML = '<div class="placeholder-msg"><div class="placeholder-icon">🔍</div><p>Searching index...</p></div>';
    paraCountBadge.textContent = 'Searching...';

    try {
      const response = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: question,
          dataset: datasetSelect.value,
          system_type: systemSelect.value,
          retrieval_type: retrievalSelect.value
        })
      });

      const data = await response.json();
      renderResults(data);
    } catch (err) {
      console.error(err);
      stepTraceContainer.innerHTML = `<div class="step-card answer"><div class="step-title">Error</div><div class="step-content">${err.message || 'Execution failed.'}</div></div>`;
    } finally {
      submitBtn.disabled = false;
      submitBtn.querySelector('.btn-text').textContent = 'Execute Reasoning';
    }
  });

  function renderResults(data) {
    executionBadge.textContent = `Completed (${data.execution_time_sec}s)`;
    executionBadge.style.color = '#10b981';

    // Render Steps
    if (data.steps && data.steps.length > 0) {
      stepTraceContainer.innerHTML = '';
      data.steps.forEach(step => {
        const stepEl = document.createElement('div');
        stepEl.className = `step-card ${step.type}`;
        
        let typeBadge = 'Thought';
        if (step.type === 'retrieval') typeBadge = 'Retrieval';
        if (step.type === 'answer') typeBadge = 'Final Answer';

        stepEl.innerHTML = `
          <div class="step-header">
            <span class="step-title">Step ${step.step_num}: ${step.title}</span>
            <span class="step-time">${step.timestamp}s</span>
          </div>
          <div class="step-content">${escapeHtml(step.content)}</div>
        `;
        stepTraceContainer.appendChild(stepEl);
      });
    }

    // Render Paragraphs
    const paras = data.retrieved_paragraphs || [];
    paraCountBadge.textContent = `${paras.length} Documents`;
    if (paras.length > 0) {
      paragraphContainer.innerHTML = '';
      paras.forEach(p => {
        const pEl = document.createElement('div');
        pEl.className = 'para-card';
        pEl.innerHTML = `
          <div class="para-title">📄 ${escapeHtml(p.title)} <span style="font-size:0.75rem; color:#9ca3af;">(Score: ${p.score})</span></div>
          <div class="para-text">${escapeHtml(p.text)}</div>
        `;
        paragraphContainer.appendChild(pEl);
      });
    } else {
      paragraphContainer.innerHTML = '<div class="placeholder-msg"><p>No external paragraphs retrieved for this setup.</p></div>';
    }
  }

  function escapeHtml(text) {
    if (!text) return '';
    return text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
