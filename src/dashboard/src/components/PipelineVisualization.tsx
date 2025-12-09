// Usage:
// npm install reactflow dagre dagre-d3-es lucide-react framer-motion
// Then import this file into your app.

import React, { useCallback, useMemo } from "react";
import ReactFlow, {
  addEdge,
  Background,
  Controls,
  Edge,
  EdgeChange,
  MiniMap,
  Node,
  NodeChange,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
  MarkerType,
  Position,
  Connection,
} from "reactflow";

import dagre from "dagre";
import "reactflow/dist/style.css";
import {
  Database,
  Download,
  Filter,
  CheckCircle,
  TrendingUp,
  BarChart3,
  Eye,
} from "lucide-react";

type PipelineStage = {
  id: string;
  label: string;
  description?: string;
  icon?: React.ReactNode;
};

const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

const nodeWidth = 320;
const nodeHeight = 140;

function getLayoutedElements(
  nodes: Node[],
  edges: Edge[],
  direction: "TB" | "LR" = "TB"
) {
  const isHorizontal = direction === "LR";
  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  const layoutedNodes = nodes.map((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };
    // let React Flow re-calc position on first render
    node.targetPosition = isHorizontal ? Position.Left : Position.Top;
    node.sourcePosition = isHorizontal ? Position.Right : Position.Bottom;
    return node;
  });

  return { nodes: layoutedNodes, edges };
}

const defaultStages: PipelineStage[] = [
  { id: "config", label: "Configuration", description: "Load trading parameters from config.toml", icon: <Database /> },
  { id: "fetch", label: "Data Fetching", description: "Download stock data from yfinance API", icon: <Download /> },
  { id: "store", label: "Data Storage", description: "Store raw data in SQLite database", icon: <Database /> },
  { id: "clean", label: "Data Cleaning", description: "Validate and preprocess data for training", icon: <Filter /> },
  { id: "train", label: "Model Training", description: "Train XGBoost model on historical data", icon: <TrendingUp /> },
  { id: "eval", label: "Model Evaluation", description: "Calculate R², RMSE, and other metrics", icon: <BarChart3 /> },
  { id: "persist", label: "Result Persistence", description: "Save trained model and predictions", icon: <CheckCircle /> },
  { id: "dash", label: "Dashboard", description: "Visualize results in React frontend", icon: <Eye /> },
];

const makeNodesAndEdges = (stages: PipelineStage[]) => {
  const nodes: Node[] = stages.map((s, i) => ({
    id: s.id,
    data: {
      label: (
        <div className="p-5 w-full h-full">
          <div className="flex flex-col gap-3 h-full">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-lg bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white shadow-lg">
                {s.icon}
              </div>
              <div className="flex-1">
                <div className="text-base font-bold text-gray-900 dark:text-gray-900">
                  {s.label}
                </div>
              </div>
            </div>
            <div className="text-sm text-gray-700 dark:text-gray-700 leading-relaxed">
              {s.description}
            </div>
          </div>
        </div>
      ),
    },
    // initial positions (will be overridden by Dagre anyway)
    position: { x: i * 200, y: i * 120 },
    style: {
      width: nodeWidth,
      height: nodeHeight,
      borderRadius: 16,
      boxShadow: "0 8px 24px rgba(0, 0, 0, 0.12)",
      border: "2px solid rgba(59, 130, 246, 0.2)",
      background: "#ffffff",
      padding: 0,
    },
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    type: "default",
  }));

  const edges: Edge[] = stages.slice(0, stages.length - 1).map((_, i) => ({
    id: `e-${stages[i].id}-${stages[i + 1].id}`,
    source: stages[i].id,
    target: stages[i + 1].id,
    markerEnd: { type: MarkerType.ArrowClosed, color: '#3b82f6' },
    animated: true,
    type: "smoothstep",
    style: { strokeWidth: 3, stroke: '#3b82f6' },
  }));

  return { nodes, edges };
};

export default function PipelineDiagram() {
  const { nodes: defaultNodes, edges: defaultEdges } = useMemo(
    () => makeNodesAndEdges(defaultStages),
    []
  );

  const layouted = useMemo(
    () => getLayoutedElements(defaultNodes, defaultEdges, "LR"),
    [defaultNodes, defaultEdges]
  );

  const [nodes, _setNodes, onNodesChange] = useNodesState(layouted.nodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layouted.edges);

  const onConnect = useCallback(
    (connection: Connection) =>
      setEdges((eds) =>
        addEdge(
          {
            ...connection,
            animated: true,
            type: "smoothstep",
            markerEnd: { type: MarkerType.Arrow },
          },
          eds
        )
      ),
    [setEdges]
  );

  const handleNodesChange = useCallback(
    (changes: NodeChange[]) => onNodesChange(changes),
    [onNodesChange]
  );

  const handleEdgesChange = useCallback(
    (changes: EdgeChange[]) => onEdgesChange(changes),
    [onEdgesChange]
  );

  return (
    <div className="h-[800px] rounded-xl border-2 border-gray-300 dark:border-gray-600 overflow-hidden bg-gray-50">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={handleNodesChange}
          onEdgesChange={handleEdgesChange}
          onConnect={onConnect}
          fitView
          snapToGrid
          snapGrid={[16, 16]}
          defaultEdgeOptions={{ animated: true }}
          panOnScroll
          zoomOnScroll
          minZoom={0.5}
          maxZoom={1.5}
          defaultViewport={{ x: 0, y: 0, zoom: 0.8 }}
        >
          <Background gap={20} color="#e5e7eb" />
          <MiniMap nodeColor={() => "#3b82f6"} maskColor="rgba(0, 0, 0, 0.1)" />
          <Controls />
        </ReactFlow>
      </ReactFlowProvider>
    </div>
  );
}