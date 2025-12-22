import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, Copy, ShieldCheck, Zap, ScanLine, AlertTriangle, Car, Info } from 'lucide-react';
import { detectAll } from '../utils/detection';
import { getSignInstruction } from '../utils/signInstructions';

const UploadMode = () => {
    const [image, setImage] = useState(null);
    const [fileObj, setFileObj] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isScanning, setIsScanning] = useState(false);
    const [plates, setPlates] = useState([]);
    const [signs, setSigns] = useState([]);
    const canvasRef = useRef(null);
    const imageRef = useRef(null);
    const containerRef = useRef(null);

    const processFile = (file) => {
        if (!file || !file.type.startsWith('image/')) return;

        const url = URL.createObjectURL(file);
        setImage(url);
        setFileObj(file);
        setPlates([]);
        setSigns([]);
        setIsScanning(true);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files?.[0]) {
            processFile(e.dataTransfer.files[0]);
        }
    };

    const handleImageLoad = async () => {
        if (!imageRef.current || !canvasRef.current) return;

        const img = imageRef.current;

        // Scan using real backend API
        const result = await detectAll(img.naturalWidth, img.naturalHeight, fileObj);
        setIsScanning(false);
        setPlates(result.plates || []);
        setSigns(result.traffic_signs || []);
    };

    // Draw detected boxes
    useEffect(() => {
        if (!canvasRef.current || !imageRef.current) return;
        if (plates.length === 0 && signs.length === 0) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const img = imageRef.current;

        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw plate detections (cyan)
        plates.forEach(det => {
            const [x, y, w, h] = det.bbox;

            ctx.strokeStyle = '#22d3ee';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            // Corner accents
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
        });

        // Draw traffic sign detections (orange)
        signs.forEach(det => {
            const [x, y, w, h] = det.bbox;

            ctx.strokeStyle = '#f97316';
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);

            // Corner accents
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
            ctx.font = 'bold 14px Outfit';
            ctx.fillStyle = '#f97316';
            ctx.fillText(det.class_name || 'SIGN', x, y - 8);
        });

    }, [plates, signs, image]);

    const reset = () => {
        setImage(null);
        setFileObj(null);
        setPlates([]);
        setSigns([]);
        setIsScanning(false);
    };

    return (
        <div className="w-full max-w-6xl mx-auto px-4">

            {!image ? (
                // Upload Area - Premium Design
                <div
                    className={`
                        relative group overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-500
                        min-h-[450px] flex flex-col items-center justify-center p-10 text-center cursor-pointer
                        ${isDragging
                            ? 'border-cyan-400 bg-cyan-950/30 scale-[1.02]'
                            : 'border-slate-700/50 hover:border-cyan-500/60 bg-slate-900/30 hover:bg-slate-900/50'}
                    `}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById('fileInput').click()}
                >
                    <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                    <input
                        id="fileInput"
                        type="file"
                        className="hidden"
                        accept="image/*"
                        onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
                    />

                    <div className="relative z-10 flex flex-col items-center">
                        <div className="bg-slate-800/80 p-5 rounded-2xl mb-6 group-hover:scale-110 group-hover:shadow-[0_0_40px_rgba(34,211,238,0.2)] transition-all duration-500 border border-slate-700/50">
                            <Upload className="w-10 h-10 text-cyan-400" />
                        </div>

                        <h2 className="text-2xl md:text-3xl font-bold text-white mb-3 tracking-wide">
                            DROP IMAGE FOR ANALYSIS
                        </h2>
                        <p className="text-slate-400 text-base max-w-md mb-6">
                            Upload a vehicle image and our AI will detect license plates and traffic signs
                        </p>

                        <div className="flex gap-3 text-xs text-slate-500 font-mono border border-slate-700/50 rounded-full px-5 py-2 bg-slate-900/70">
                            <span>.JPG</span>
                            <span className="text-slate-700">•</span>
                            <span>.PNG</span>
                            <span className="text-slate-700">•</span>
                            <span>.WEBP</span>
                        </div>
                    </div>
                </div>
            ) : (
                // Result View - Split Layout
                <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">

                    {/* Main Image View - 3 columns */}
                    <div className="lg:col-span-3 relative glass-card rounded-2xl overflow-hidden">
                        <div className="relative" ref={containerRef}>
                            <img
                                ref={imageRef}
                                src={image}
                                alt="Uploaded target"
                                className="w-full h-auto block"
                                onLoad={handleImageLoad}
                            />

                            <canvas
                                ref={canvasRef}
                                className="absolute inset-0 w-full h-full pointer-events-none"
                            />

                            {/* Scanning Overlay */}
                            {isScanning && (
                                <div className="absolute inset-0 bg-slate-950/70 backdrop-blur-sm flex items-center justify-center flex-col gap-4">
                                    <div className="absolute inset-0 overflow-hidden">
                                        <div className="absolute left-0 right-0 h-1 bg-gradient-to-b from-transparent via-cyan-400 to-transparent animate-scanline" />
                                    </div>
                                    <div className="relative z-10 flex flex-col items-center">
                                        <div className="w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-400 rounded-full animate-spin" />
                                        <p className="mt-4 text-cyan-400 font-semibold tracking-widest text-sm uppercase">Analyzing Image...</p>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Close Button */}
                        <button onClick={reset} className="absolute top-3 right-3 bg-slate-900/80 hover:bg-red-500/80 text-white p-2 rounded-lg backdrop-blur border border-slate-700/50 shadow-lg transition-colors">
                            <X className="w-5 h-5" />
                        </button>
                    </div>

                    {/* Sidebar Results - 2 columns */}
                    <div className="lg:col-span-2 space-y-4">
                        <div className="glass-panel rounded-xl p-4">
                            <h3 className="text-lg font-bold text-white flex items-center gap-2 mb-1">
                                <ShieldCheck className="w-5 h-5 text-cyan-400" />
                                Detection Results
                            </h3>
                            <p className="text-xs text-slate-500">Powered by YOLO + EasyOCR</p>
                        </div>

                        {isScanning ? (
                            <div className="space-y-3">
                                {[1, 2].map(i => (
                                    <div key={i} className="h-28 glass-card rounded-xl animate-pulse" />
                                ))}
                            </div>
                        ) : (
                            <>
                                {/* License Plates */}
                                {plates.length > 0 && (
                                    <div className="space-y-3">
                                        <h4 className="text-sm font-semibold text-cyan-400 flex items-center gap-2">
                                            <Car className="w-4 h-4" />
                                            License Plates ({plates.length})
                                        </h4>
                                        {plates.map((res) => (
                                            <div key={res.id} className="glass-card rounded-xl p-4 hover:border-cyan-500/50 transition-all group">
                                                {res.crop_image && (
                                                    <div className="mb-3 p-2 bg-black/50 rounded-lg border border-slate-700/50 flex items-center justify-center">
                                                        <img
                                                            src={res.crop_image}
                                                            alt="Detected Plate Crop"
                                                            className="max-h-20 object-contain rounded shadow-lg"
                                                        />
                                                    </div>
                                                )}

                                                <div className="flex justify-between items-start mb-2">
                                                    <span className="text-xs font-mono text-slate-500">PLATE #{res.id + 1}</span>
                                                    <span className="text-xs font-bold text-emerald-400 px-2 py-0.5 bg-emerald-500/10 rounded-full border border-emerald-500/20">
                                                        {(res.confidence * 100).toFixed(0)}% OCR
                                                    </span>
                                                </div>

                                                <div className="flex items-center gap-3">
                                                    <div className="flex-1 bg-gradient-to-r from-slate-100 to-slate-200 text-slate-900 font-bold font-mono px-4 py-2 rounded-lg border-2 border-slate-400 text-xl tracking-widest shadow-inner text-center">
                                                        {res.text || "N/A"}
                                                    </div>
                                                    <button
                                                        className="p-2.5 hover:bg-cyan-500/20 rounded-lg text-slate-400 hover:text-cyan-400 transition-colors border border-transparent hover:border-cyan-500/30"
                                                        title="Copy text"
                                                        onClick={() => navigator.clipboard.writeText(res.text)}
                                                    >
                                                        <Copy className="w-4 h-4" />
                                                    </button>
                                                </div>

                                                <div className="mt-3 text-xs text-slate-500 flex items-center gap-2">
                                                    <Zap className="w-3 h-3 text-cyan-500" />
                                                    Detection Conf: {(res.detection_conf * 100).toFixed(0)}%
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* Traffic Signs */}
                                {signs.length > 0 && (
                                    <div className="space-y-3">
                                        <h4 className="text-sm font-semibold text-orange-400 flex items-center gap-2">
                                            <AlertTriangle className="w-4 h-4" />
                                            Traffic Signs ({signs.length})
                                        </h4>
                                        {signs.map((res) => {
                                            const signInfo = getSignInstruction(res.class_name);
                                            return (
                                                <div key={res.id} className="glass-card rounded-xl p-4 hover:border-orange-500/50 transition-all group">
                                                    {res.crop_image && (
                                                        <div className="mb-3 p-2 bg-black/50 rounded-lg border border-slate-700/50 flex items-center justify-center">
                                                            <img
                                                                src={res.crop_image}
                                                                alt="Detected Sign Crop"
                                                                className="max-h-20 object-contain rounded shadow-lg"
                                                            />
                                                        </div>
                                                    )}

                                                    <div className="flex justify-between items-center mb-2">
                                                        <span className="text-lg font-bold text-white">{signInfo.displayName}</span>
                                                        <span className="text-xs font-bold text-orange-400 px-2 py-0.5 bg-orange-500/10 rounded-full border border-orange-500/20">
                                                            {(res.confidence * 100).toFixed(0)}%
                                                        </span>
                                                    </div>

                                                    {/* User-friendly instruction */}
                                                    <div className="mt-2 p-3 bg-orange-500/10 border border-orange-500/20 rounded-lg">
                                                        <div className="flex items-start gap-2">
                                                            <Info className="w-4 h-4 text-orange-400 flex-shrink-0 mt-0.5" />
                                                            <p className="text-sm text-orange-200 leading-relaxed">
                                                                {signInfo.instruction}
                                                            </p>
                                                        </div>
                                                    </div>
                                                </div>
                                            );
                                        })}
                                    </div>
                                )}

                                {/* No Results */}
                                {plates.length === 0 && signs.length === 0 && (
                                    <div className="text-center p-8 glass-card rounded-xl text-slate-500">
                                        <ScanLine className="w-10 h-10 mx-auto mb-3 text-slate-600" />
                                        <p className="font-medium">No detections found</p>
                                        <p className="text-xs mt-1">Try an image with visible plates or traffic signs</p>
                                    </div>
                                )}
                            </>
                        )}
                    </div>

                </div>
            )}
        </div>
    );
}

export default UploadMode;
