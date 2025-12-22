import React, { useRef, useEffect, useState } from 'react';
import { Camera, AlertTriangle, PlayCircle, StopCircle, Cpu, Radio, Signal, Car, Info } from 'lucide-react';
import { detectAll } from '../utils/detection';
import { getSignInstruction } from '../utils/signInstructions';

const CameraMode = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [error, setError] = useState(null);
    const [detectedPlate, setDetectedPlate] = useState(null);
    const [detectedSigns, setDetectedSigns] = useState([]);
    const [plateHistory, setPlateHistory] = useState([]);
    const [signHistory, setSignHistory] = useState([]);

    // Real detection loop
    useEffect(() => {
        let interval;
        if (isStreaming) {
            interval = setInterval(async () => {
                if (videoRef.current && canvasRef.current) {
                    const video = videoRef.current;
                    const canvas = canvasRef.current;

                    // Create a temporary canvas to capture the frame
                    const tempCanvas = document.createElement('canvas');
                    tempCanvas.width = video.videoWidth;
                    tempCanvas.height = video.videoHeight;
                    const tempCtx = tempCanvas.getContext('2d');
                    tempCtx.drawImage(video, 0, 0);

                    // Convert to blob to send to API
                    tempCanvas.toBlob(async (blob) => {
                        if (!blob) return;

                        const result = await detectAll(video.videoWidth, video.videoHeight, blob);
                        const plates = result.plates || [];
                        const signs = result.traffic_signs || [];

                        // Draw results
                        const ctx = canvas.getContext('2d');
                        canvas.width = video.videoWidth;
                        canvas.height = video.videoHeight;
                        ctx.clearRect(0, 0, canvas.width, canvas.height);

                        // Draw plate detections
                        if (plates.length > 0) {
                            const det = plates[0];
                            setDetectedPlate(det);

                            // Add to history if new
                            if (det.text && det.text !== plateHistory[0]?.text) {
                                setPlateHistory(prev => [det, ...prev].slice(0, 5));
                            }

                            const [x, y, w, h] = det.bbox;

                            // Draw targeting box (cyan)
                            ctx.strokeStyle = '#22d3ee';
                            ctx.lineWidth = 3;
                            ctx.strokeRect(x, y, w, h);

                            // Corner brackets
                            ctx.fillStyle = '#22d3ee';
                            const cornerSize = Math.min(w * 0.15, 20);
                            const thickness = 4;
                            ctx.fillRect(x, y, cornerSize, thickness);
                            ctx.fillRect(x, y, thickness, cornerSize);
                            ctx.fillRect(x + w - cornerSize, y, cornerSize, thickness);
                            ctx.fillRect(x + w - thickness, y, thickness, cornerSize);
                            ctx.fillRect(x, y + h - thickness, cornerSize, thickness);
                            ctx.fillRect(x, y + h - cornerSize, thickness, cornerSize);
                            ctx.fillRect(x + w - cornerSize, y + h - thickness, cornerSize, thickness);
                            ctx.fillRect(x + w - thickness, y + h - cornerSize, thickness, cornerSize);
                        } else {
                            setDetectedPlate(null);
                        }

                        // Draw sign detections (orange)
                        setDetectedSigns(signs);
                        signs.forEach(det => {
                            const [x, y, w, h] = det.bbox;

                            ctx.strokeStyle = '#f97316';
                            ctx.lineWidth = 3;
                            ctx.strokeRect(x, y, w, h);

                            // Corner brackets
                            ctx.fillStyle = '#f97316';
                            const cornerSize = Math.min(w * 0.15, 20);
                            const thickness = 4;
                            ctx.fillRect(x, y, cornerSize, thickness);
                            ctx.fillRect(x, y, thickness, cornerSize);
                            ctx.fillRect(x + w - cornerSize, y, cornerSize, thickness);
                            ctx.fillRect(x + w - thickness, y, thickness, cornerSize);
                            ctx.fillRect(x, y + h - thickness, cornerSize, thickness);
                            ctx.fillRect(x, y + h - cornerSize, thickness, cornerSize);
                            ctx.fillRect(x + w - cornerSize, y + h - thickness, cornerSize, thickness);
                            ctx.fillRect(x + w - thickness, y + h - cornerSize, thickness, cornerSize);

                            // Label
                            ctx.font = 'bold 12px Outfit';
                            ctx.fillStyle = '#f97316';
                            ctx.fillText(det.class_name || 'SIGN', x, y - 5);

                            // Add to sign history if new
                            if (det.class_name && !signHistory.find(s => s.class_name === det.class_name)) {
                                setSignHistory(prev => [det, ...prev].slice(0, 5));
                            }
                        });

                    }, 'image/jpeg', 0.8);
                }
            }, 1500); // 1.5s interval to avoid overloading backend
        }
        return () => clearInterval(interval);
    }, [isStreaming, plateHistory, signHistory]);

    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setIsStreaming(true);
                setError(null);
            }
        } catch (err) {
            setError("Camera access denied. Please check permissions.");
            console.error(err);
        }
    };

    const stopCamera = () => {
        if (videoRef.current?.srcObject) {
            videoRef.current.srcObject.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
            setIsStreaming(false);
            setDetectedPlate(null);
            setDetectedSigns([]);
        }
    };

    // Cleanup on unmount
    useEffect(() => {
        return () => stopCamera();
    }, []);

    return (
        <div className="max-w-5xl mx-auto px-4">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">

                {/* Main Video Feed */}
                <div className="lg:col-span-3 relative rounded-2xl overflow-hidden glass-card aspect-video group">

                    {!isStreaming && !error && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-slate-950/90 gap-6">
                            <div className="p-5 rounded-2xl bg-slate-800/80 border border-slate-700/50 shadow-xl shadow-cyan-500/10">
                                <Camera className="w-14 h-14 text-cyan-400" />
                            </div>
                            <div className="text-center">
                                <h3 className="text-xl font-bold text-white mb-2">Live Detection Feed</h3>
                                <p className="text-slate-400 mb-6 text-sm">Connect webcam for real-time plate & sign detection</p>
                                <button
                                    onClick={startCamera}
                                    className="px-8 py-3 bg-gradient-to-r from-cyan-500 to-cyan-600 hover:from-cyan-400 hover:to-cyan-500 text-white font-bold rounded-xl transition-all shadow-lg shadow-cyan-500/30 flex items-center gap-2 mx-auto"
                                >
                                    <PlayCircle className="w-5 h-5" />
                                    Activate Scanner
                                </button>
                            </div>
                        </div>
                    )}

                    {error && (
                        <div className="absolute inset-0 flex flex-col items-center justify-center z-10 bg-slate-950/90">
                            <AlertTriangle className="w-14 h-14 text-rose-500 mb-4" />
                            <p className="text-lg text-white font-bold">{error}</p>
                            <button onClick={startCamera} className="mt-4 px-6 py-2 bg-slate-800 rounded-lg text-white hover:bg-slate-700 transition-colors">Retry</button>
                        </div>
                    )}

                    <video
                        ref={videoRef}
                        autoPlay
                        playsInline
                        muted
                        className="w-full h-full object-cover"
                    />
                    <canvas
                        ref={canvasRef}
                        className="absolute inset-0 w-full h-full pointer-events-none"
                    />

                    {/* HUD Overlay */}
                    {isStreaming && (
                        <>
                            {/* Top Bar */}
                            <div className="absolute top-0 left-0 right-0 p-3 flex justify-between items-center bg-gradient-to-b from-black/60 to-transparent">
                                <div className="flex items-center gap-2 text-xs text-cyan-400 font-mono">
                                    <div className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                                    <span>REC</span>
                                </div>
                                <div className="flex items-center gap-3 text-xs text-slate-400 font-mono">
                                    <Signal className="w-4 h-4 text-green-400" />
                                    <span>AI ACTIVE</span>
                                </div>
                            </div>

                            {/* Detection Result Overlay - Plate */}
                            {detectedPlate && detectedPlate.text && (
                                <div className="absolute bottom-20 left-1/2 -translate-x-1/2 glass-panel px-5 py-3 rounded-xl flex items-center gap-4 animate-in fade-in zoom-in-95 duration-300">
                                    {detectedPlate.crop_image && (
                                        <img src={detectedPlate.crop_image} alt="Plate" className="h-10 rounded border border-slate-600" />
                                    )}
                                    <div>
                                        <p className="text-xs text-cyan-400 uppercase tracking-wider font-semibold">Plate Detected</p>
                                        <p className="text-xl font-mono font-bold text-white tracking-widest">{detectedPlate.text}</p>
                                    </div>
                                </div>
                            )}

                            {/* Detection Result Overlay - Signs */}
                            {detectedSigns.length > 0 && (
                                <div className="absolute top-16 right-3 glass-panel px-4 py-3 rounded-xl max-w-xs">
                                    <p className="text-xs text-orange-400 uppercase tracking-wider font-semibold mb-2">Signs Detected</p>
                                    <div className="space-y-2">
                                        {detectedSigns.map((sign, i) => {
                                            const signInfo = getSignInstruction(sign.class_name);
                                            return (
                                                <div key={i} className="bg-orange-500/20 px-3 py-2 rounded-lg">
                                                    <span className="text-sm font-bold text-white block">{signInfo.displayName}</span>
                                                    <span className="text-xs text-orange-200 block mt-1">{signInfo.instruction}</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>
                            )}
                        </>
                    )}

                    {/* Controls */}
                    {isStreaming && (
                        <div className="absolute bottom-4 left-0 right-0 flex justify-center gap-3 opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                            <button
                                onClick={stopCamera}
                                className="bg-red-500/80 hover:bg-red-500 text-white px-5 py-2.5 rounded-full backdrop-blur flex items-center gap-2 font-medium transition-all text-sm"
                            >
                                <StopCircle className="w-4 h-4" />
                                Stop
                            </button>
                        </div>
                    )}
                </div>

                {/* Side Panel - History & Status */}
                <div className="lg:col-span-1 space-y-4">
                    {/* Status Card */}
                    <div className="glass-panel rounded-xl p-4">
                        <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                            <Cpu className="w-4 h-4 text-cyan-400" />
                            System Status
                        </h4>
                        <div className="space-y-2 text-xs">
                            <div className="flex items-center gap-2">
                                <div className={`w-2 h-2 rounded-full ${isStreaming ? 'bg-green-500 animate-pulse' : 'bg-slate-500'}`} />
                                <span className="text-slate-300">Camera: {isStreaming ? 'Active' : 'Standby'}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-cyan-500" />
                                <span className="text-slate-300">Plate Detection: On</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <div className="w-2 h-2 rounded-full bg-orange-500" />
                                <span className="text-slate-300">Sign Detection: On</span>
                            </div>
                        </div>
                    </div>

                    {/* Plate History Card */}
                    <div className="glass-panel rounded-xl p-4">
                        <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                            <Car className="w-4 h-4 text-cyan-400" />
                            Recent Plates
                        </h4>
                        {plateHistory.length > 0 ? (
                            <div className="space-y-2">
                                {plateHistory.map((item, i) => (
                                    <div key={i} className="flex items-center gap-2 p-2 bg-slate-800/50 rounded-lg border border-slate-700/50">
                                        {item.crop_image && (
                                            <img src={item.crop_image} alt="" className="h-6 rounded" />
                                        )}
                                        <span className="font-mono text-sm text-white flex-1">{item.text}</span>
                                    </div>
                                ))}
                            </div>
                        ) : (
                            <p className="text-xs text-slate-500 text-center py-4">No plates detected yet</p>
                        )}
                    </div>

                    {/* Sign History Card */}
                    <div className="glass-panel rounded-xl p-4">
                        <h4 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
                            <AlertTriangle className="w-4 h-4 text-orange-400" />
                            Recent Signs
                        </h4>
                        {signHistory.length > 0 ? (
                            <div className="space-y-2">
                                {signHistory.map((item, i) => {
                                    const signInfo = getSignInstruction(item.class_name);
                                    return (
                                        <div key={i} className="p-2 bg-orange-500/10 border border-orange-500/30 rounded-lg" title={signInfo.instruction}>
                                            <span className="text-xs font-semibold text-orange-400 block">{signInfo.displayName}</span>
                                            <span className="text-xs text-orange-200/70 line-clamp-2">{signInfo.instruction}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        ) : (
                            <p className="text-xs text-slate-500 text-center py-4">No signs detected yet</p>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CameraMode;
