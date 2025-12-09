/**
 * WebSocket Service for Real-Time Agent Updates
 * Connects to Flask-SocketIO backend for live analysis streaming
 */

import { io, Socket } from 'socket.io-client';

const SOCKET_URL = process.env.REACT_APP_API_URL?.replace('/api', '') || 'http://localhost:8000';

export interface AnalysisUpdate {
  symbol: string;
  agent: string;
  timestamp: string;
  current_price?: number;
  predicted_price?: number;
  change_pct?: number;
  signal: string;
  action: string;
  confidence: number;
  analysis?: any;
  decision?: any;
}

class WebSocketService {
  private socket: Socket | null = null;
  private connected: boolean = false;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private listeners: Map<string, Set<Function>> = new Map();

  constructor() {
    this.connect();
  }

  /**
   * Connect to WebSocket server
   */
  connect(): void {
    if (this.socket?.connected) {
      console.log('[WebSocket] Already connected');
      return;
    }

    console.log('[WebSocket] Connecting to:', SOCKET_URL);

    this.socket = io(SOCKET_URL, {
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: this.maxReconnectAttempts,
    });

    this.setupEventHandlers();
  }

  /**
   * Setup socket event handlers
   */
  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('[WebSocket] Connected successfully');
      this.connected = true;
      this.reconnectAttempts = 0;
      this.emit('connection_status', { connected: true });
    });

    this.socket.on('disconnect', () => {
      console.log('[WebSocket] Disconnected');
      this.connected = false;
      this.emit('connection_status', { connected: false });
    });

    this.socket.on('connect_error', (error: Error) => {
      console.error('[WebSocket] Connection error:', error.message);
      this.reconnectAttempts++;

      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        console.error('[WebSocket] Max reconnection attempts reached');
        this.emit('connection_error', { error: 'Max reconnection attempts reached' });
      }
    });

    this.socket.on('connection_response', (data: any) => {
      console.log('[WebSocket] Connection response:', data);
    });

    // Real-time analysis updates from background agents
    this.socket.on('analysis_update', (data: AnalysisUpdate) => {
      console.log('[WebSocket] Analysis update:', data);
      this.emit('analysis_update', data);
    });

    // Analysis results from on-demand requests
    this.socket.on('analysis_result', (data: AnalysisUpdate) => {
      console.log('[WebSocket] Analysis result:', data);
      this.emit('analysis_result', data);
    });
  }

  /**
   * Subscribe to events
   */
  on(event: string, callback: Function): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  /**
   * Unsubscribe from events
   */
  off(event: string, callback: Function): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.delete(callback);
    }
  }

  /**
   * Emit event to all listeners
   */
  private emit(event: string, data: any): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`[WebSocket] Error in listener for ${event}:`, error);
        }
      });
    }
  }

  /**
   * Request analysis for a specific symbol and agent
   */
  requestAnalysis(symbol: string, agent: string = 'agentVD'): void {
    if (!this.socket?.connected) {
      console.warn('[WebSocket] Not connected. Cannot request analysis.');
      return;
    }

    console.log(`[WebSocket] Requesting analysis for ${symbol} with ${agent}`);
    this.socket.emit('request_analysis', { symbol, agent });
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.connected && this.socket?.connected === true;
  }

  /**
   * Disconnect from server
   */
  disconnect(): void {
    if (this.socket) {
      console.log('[WebSocket] Disconnecting...');
      this.socket.disconnect();
      this.socket = null;
      this.connected = false;
    }
  }

  /**
   * Reconnect to server
   */
  reconnect(): void {
    this.disconnect();
    this.reconnectAttempts = 0;
    this.connect();
  }
}

// Export singleton instance
export const wsService = new WebSocketService();

export default wsService;
