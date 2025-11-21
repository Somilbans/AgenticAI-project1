const TabsBar = ({ active, onChange }) => (
  <div className="tabs">
    <button
      className={active === "playground" ? "active" : ""}
      onClick={() => onChange("playground")}
    >
      Playground
    </button>
    <button
      className={active === "insights" ? "active" : ""}
      onClick={() => onChange("insights")}
    >
      Insights
    </button>
  </div>
);

window.TabsBar = TabsBar;

