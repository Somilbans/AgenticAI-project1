const { useState } = React;

const PlaygroundTab = ({
  collections,
  onSubmit,
  loading,
  command,
  error,
  answer,
}) => {
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("collection"); // "collection" or "intelligent"
  const [collection, setCollection] = useState(collections[0]?.value || "onbench");
  const [topK, setTopK] = useState("3");

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit({ question, mode, collection, topK });
  };

  return (
    <>
      <p className="hint">
        Choose a query mode: Collection Based searches a specific collection, Intelligent System uses LLM to understand and match employees to positions. Set the top-k value and we'll craft the CLI command and run the query for you.
      </p>
      <form onSubmit={handleSubmit}>
        <label>Query Mode</label>
        <select value={mode} onChange={(e) => setMode(e.target.value)}>
          <option value="collection">Collection Based</option>
          <option value="intelligent">Intelligent System</option>
        </select>

        {mode === "collection" && (
          <>
            <label>Collection</label>
            <select value={collection} onChange={(e) => setCollection(e.target.value)}>
              {collections.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </>
        )}

        <label>Question</label>
        <textarea
          rows="3"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={mode === "intelligent" 
            ? "Ask to match employees to positions or vice versa (e.g., 'Find jobs for John' or 'Who can fill Python developer role?')..."
            : "Ask anything about the bench or positions..."}
        />

        <label>Top-k</label>
        <div className="topk-control">
          <input
            type="range"
            min="1"
            max="15"
            value={Number(topK) || 3}
            onChange={(e) => setTopK(String(e.target.value))}
          />
          <input
            type="number"
            min="1"
            max="50"
            value={topK}
            onChange={(e) => setTopK(e.target.value)}
          />
        </div>

        <button type="submit" disabled={loading}>
          {loading ? "Querying..." : "Generate & Query"}
        </button>
      </form>

      {command && (
        <div className="result">
          <strong>CLI Command</strong>
          <div>{command}</div>
        </div>
      )}

      {error && (
        <div className="result error-box">
          <strong>Error</strong>
          <div>{error}</div>
        </div>
      )}

      {answer && (
        <div className="result">
          <strong>Answer</strong>
          <div>{answer}</div>
        </div>
      )}
    </>
  );
};

window.PlaygroundTab = PlaygroundTab;

