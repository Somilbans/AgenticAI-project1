const { useEffect, useRef } = React;

const InsightsTab = ({ insights, loading, error }) => {
  const chartsRef = useRef({});

  useEffect(() => {
    if (!loading && insights) {
      renderCharts(insights, chartsRef);
    }
  }, [loading, insights]);

  if (loading) {
    return <p className="hint">Loading insights…</p>;
  }

  if (error) {
    return (
      <div className="result error-box">
        <strong>Error</strong>
        <div>{error}</div>
      </div>
    );
  }

  if (!insights) {
    return null;
  }

  return (
    <div className="insight-panel">
      <div className="chart-group">
        <h2>Bench Insights</h2>
        <div className="charts">
          <div className="chart-card">
            <h3>Bench Experience Distribution</h3>
            <canvas id="benchExpChart"></canvas>
          </div>
          <div className="chart-card">
            <h3>Bench Skill Distribution</h3>
            <canvas id="benchSkillDistChart"></canvas>
          </div>
        </div>
      </div>

      <div className="chart-group">
        <h2>Positions Insights</h2>
        <div className="charts">
          <div className="chart-card">
            <h3>Experience Demand</h3>
            <canvas id="positionExpChart"></canvas>
          </div>
          <div className="chart-card">
            <h3>Location vs Avg Experience</h3>
            <canvas id="locationChart"></canvas>
          </div>
          <div className="chart-card">
            <h3>Grade vs Experience</h3>
            <canvas id="gradeChart"></canvas>
          </div>
          <div className="chart-card">
            <h3>Skill Distribution (Positions)</h3>
            <canvas id="skillDistChart"></canvas>
          </div>
        </div>
      </div>

    </div>
  );
};

function renderCharts(insights, chartsRef) {
  const benchData = insights.bench || {};
  const positionsData = insights.positions || {};

  const destroyChart = (id) => {
    if (chartsRef.current[id]) {
      chartsRef.current[id].destroy();
      chartsRef.current[id] = null;
    }
  };

  const ensureChart = (id, config) => {
    destroyChart(id);
    const canvas = document.getElementById(id);
    if (canvas) {
      chartsRef.current[id] = new Chart(canvas.getContext("2d"), config);
    }
  };

  const axisTitle = (text, subtitle) => ({
    title: {
      display: true,
      text: subtitle ? `${text} (${subtitle})` : text,
      color: "#0f172a",
    },
    ticks: { color: "#0f172a" },
  });

  if (benchData.experience_distribution) {
    ensureChart("benchExpChart", {
      type: "bar",
      data: {
        labels: Object.keys(benchData.experience_distribution),
        datasets: [
          {
            label: "Employees",
            data: Object.values(benchData.experience_distribution),
            backgroundColor: "#2563eb",
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: axisTitle("Experience Range", "years"),
          y: axisTitle("Employees", "count"),
        },
      },
    });
  }

  if (benchData.skill_distribution) {
    ensureChart("benchSkillDistChart", {
      type: "bar",
      data: {
        labels: benchData.skill_distribution.map((item) => item.skill),
        datasets: [
          {
            label: "Mentions",
            data: benchData.skill_distribution.map((item) => item.count),
            backgroundColor: "#10b981",
          },
        ],
      },
      options: {
        responsive: true,
        indexAxis: "y",
        scales: {
          x: axisTitle("Mentions", "bench top"),
          y: axisTitle("Skill"),
        },
      },
    });
  }

  if (positionsData.experience_distribution) {
    ensureChart("positionExpChart", {
      type: "bar",
      data: {
        labels: Object.keys(positionsData.experience_distribution),
        datasets: [
          {
            label: "Requisitions",
            data: Object.values(positionsData.experience_distribution),
            backgroundColor: "#7c3aed",
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
          x: axisTitle("Experience Range", "years"),
          y: axisTitle("Open Roles", "count"),
        },
      },
    });
  }

  if (positionsData.location_experience) {
    const labels = positionsData.location_experience.map((item) => item.label);
    const minVals = positionsData.location_experience.map((item) => item.avg_min);
    const maxVals = positionsData.location_experience.map((item) => item.avg_max);
    ensureChart("locationChart", {
      type: "bar",
      data: {
        labels,
        datasets: [
          { label: "Avg Min", data: minVals, backgroundColor: "#06b6d4" },
          { label: "Avg Max", data: maxVals, backgroundColor: "#0ea5e9" },
        ],
      },
      options: {
        responsive: true,
        scales: {
          x: axisTitle("Location"),
          y: axisTitle("Years", "avg required"),
        },
      },
    });
  }

  if (positionsData.grade_experience) {
    const labels = positionsData.grade_experience.map((item) => item.grade);
    const minVals = positionsData.grade_experience.map((item) => item.avg_min);
    const maxVals = positionsData.grade_experience.map((item) => item.avg_max);
    ensureChart("gradeChart", {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Avg Min", data: minVals, borderColor: "#a855f7", fill: false },
          { label: "Avg Max", data: maxVals, borderColor: "#3b82f6", fill: false },
        ],
      },
      options: {
        responsive: true,
        scales: {
          x: axisTitle("Grade"),
          y: axisTitle("Years", "avg required"),
        },
      },
    });
  }

  if (positionsData.skill_distribution) {
    ensureChart("skillDistChart", {
      type: "bar",
      data: {
        labels: positionsData.skill_distribution.map((item) => item.skill),
        datasets: [
          {
            label: "Mentions",
            data: positionsData.skill_distribution.map((item) => item.count),
            backgroundColor: "#f97316",
          },
        ],
      },
      options: {
        responsive: true,
        indexAxis: "y",
        scales: {
          x: axisTitle("Mentions", "positions top"),
          y: axisTitle("Skill"),
        },
      },
    });
  }
}

window.InsightsTab = InsightsTab;

