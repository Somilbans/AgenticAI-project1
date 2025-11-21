const { useState, useEffect } = React;
const API_BASE =
  window.API_BASE ||
  (window.location.origin.startsWith("file")
    ? "http://127.0.0.1:5000"
    : window.location.origin);

const COLLECTION_OPTIONS = [
  { label: "Bench", value: "onbench" },
  { label: "Positions", value: "availablepositions" },
];

function App() {
  const [command, setCommand] = useState("");
  const [answer, setAnswer] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("playground");
  const [insights, setInsights] = useState(null);
  const [insightError, setInsightError] = useState("");
  const [insightLoading, setInsightLoading] = useState(false);

  const handleQuery = async ({ question, collection, topK }) => {
    const trimmedQuestion = (question || "").trim();
    const topKValue = Number(topK) || 3;

    if (!trimmedQuestion) {
      alert("Please enter a question.");
      return;
    }

    const cliParts = [
      "python main.py ask",
      `"${trimmedQuestion}"`,
      `--collection "${collection}"`,
      `--top-k ${topKValue}`,
    ];
    setCommand(cliParts.join(" "));

    setLoading(true);
    setAnswer(null);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: trimmedQuestion,
          collection,
          top_k: topKValue,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || "Query failed");
      }
      setAnswer(data.answer);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (activeTab !== "insights" || insights || insightLoading) {
      return;
    }
    setInsightLoading(true);
    setInsightError("");
    fetch(`${API_BASE}/api/insights`)
      .then((res) => res.json())
      .then((data) => setInsights(data))
      .catch((err) => setInsightError(err.message))
      .finally(() => setInsightLoading(false));
  }, [activeTab, insights, insightLoading]);

  return (
    <div className="card">
      <h1>Bench Intelligence Hub</h1>
      <TabsBar active={activeTab} onChange={setActiveTab} />

      {activeTab === "playground" ? (
        <PlaygroundTab
          collections={COLLECTION_OPTIONS}
          onSubmit={handleQuery}
          loading={loading}
          command={command}
          error={error}
          answer={answer}
        />
      ) : (
        <InsightsTab
          insights={insights}
          loading={insightLoading}
          error={insightError}
        />
      )}
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

