const ctx = document.getElementById('forecastChart').getContext('2d');
let forecastChart;

let weatherData = null;

// Initialize UI
async function initDashboard() {
    try {
        const response = await fetch('latest_forecast.json');
        weatherData = await response.json();
        
        const districtNames = Object.keys(weatherData.districts).sort();
        
        const nav = document.getElementById('district-nav');
        nav.innerHTML = districtNames.map(dist => `
            <div class="district-item ${dist === weatherData.seed_district ? 'selected' : ''}" onclick="selectDistrict('${dist}')">
                ${dist}
            </div>
        `).join('');

        document.getElementById('current-time').innerText = `Sync: ${weatherData.timestamp}`;
        selectDistrict(weatherData.seed_district);
    } catch (e) {
        console.error("Dashboard error: Ensure you ran predict_live.py first", e);
        renderChart(generateDummyData());
    }
}

function selectDistrict(name) {
    if (!weatherData) return;
    
    document.querySelectorAll('.district-item').forEach(item => {
        item.classList.remove('selected');
        if (item.innerText === name) item.classList.add('selected');
    });

    const data = weatherData.districts[name];
    document.getElementById('current-district').innerText = name;
    document.getElementById('main-temp').innerHTML = `${data.current.temp.toFixed(1)}<span>°C</span>`;
    document.getElementById('main-hum').innerText = `${data.current.hum.toFixed(1)}%`;
    document.getElementById('main-pres').innerText = "1010 hPa"; // Baseline
    
    // Update Chart
    const forecast = {
        temps: data.forecast.map(f => f.t),
        hums: data.forecast.map(f => f.h)
    };
    renderChart(forecast);
}

function renderChart(data) {
    if (forecastChart) forecastChart.destroy();
    
    forecastChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['+1h', '+2h', '+3h', '+4h', '+5h', '+6h'],
            datasets: [{
                label: 'Temperature (°C)',
                data: data.temps,
                borderColor: '#00d2ff',
                backgroundColor: 'rgba(0, 210, 255, 0.1)',
                fill: true,
                tension: 0.4,
                pointRadius: 5,
                pointBackgroundColor: '#00d2ff'
            }, {
                label: 'Humidity (%)',
                data: data.hums,
                borderColor: '#9d50bb',
                borderDash: [5, 5],
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: 'rgba(255, 255, 255, 0.7)', font: { family: 'Outfit' } } }
            },
            scales: {
                x: { ticks: { color: 'rgba(255, 255, 255, 0.5)' }, grid: { display: false } },
                y: { ticks: { color: 'rgba(255, 255, 255, 0.5)' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }
            }
        }
    });
}

function generateDummyData() {
    return {
        temps: [36, 35.5, 34.2, 33.1, 32.4, 31.8],
        hums: [26, 28, 32, 35, 40, 42]
    };
}

initDashboard();
