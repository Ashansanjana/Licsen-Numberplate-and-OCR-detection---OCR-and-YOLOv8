import React, { useState, useRef } from 'react';
import { Video, Upload, X, Loader2, Download, Play, AlertTriangle, Car, FileVideo } from 'lucide-react';

const VideoMode = () => {
    const [video, setVideo] = useState(null);
    const [videoFile, setVideoFile] = useState(null);
    const [isDragging, setIsDragging] = useState(false);
    const [isProcessing, setIsProcessing] = useState(false);
    const [progress, setProgress] = useState(0);
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const fileInputRef = useRef(null);

    const processFile = (file) => {
        if (!file) return;

        const validTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/webm', 'video/quicktime'];
        if (!validTypes.some(type => file.type.includes(type.split('/')[1]))) {
            setError('Please upload a valid video file (MP4, AVI, MOV, WebM)');
            return;
        }

        const url = URL.createObjectURL(file);
        setVideo(url);
        setVideoFile(file);
        setResult(null);
        setError(null);
    };

    const handleDrop = (e) => {
        e.preventDefault();
        setIsDragging(false);
        if (e.dataTransfer.files?.[0]) {
            processFile(e.dataTransfer.files[0]);
        }
    };

    const startProcessing = async () => {
        if (!videoFile) return;

        setIsProcessing(true);
        setProgress(0);
        setError(null);

        // Simulate progress while waiting for backend
        const progressInterval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 90) return prev;
                return prev + Math.random() * 10;
            });
        }, 500);

        try {
            const formData = new FormData();
            formData.append('file', videoFile);

            const response = await fetch('http://localhost:8000/detect/video', {
                method: 'POST',
                body: formData,
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                throw new Error(`Processing failed: ${response.statusText}`);
            }

            const data = await response.json();
            setProgress(100);
            setResult(data);

        } catch (err) {
            setError(err.message || 'Video processing failed');
            console.error('Video processing error:', err);
        } finally {
            clearInterval(progressInterval);
            setIsProcessing(false);
        }
    };

    const reset = () => {
        setVideo(null);
        setVideoFile(null);
        setResult(null);
        setError(null);
        setProgress(0);
        setIsProcessing(false);
    };

    const downloadVideo = () => {
        if (result?.download_url) {
            window.open(`http://localhost:8000${result.download_url}`, '_blank');
        }
    };

    return (
        <div className="w-full max-w-6xl mx-auto px-4">
            {!video ? (
                // Upload Area
                <div
                    className={`
                        relative group overflow-hidden rounded-2xl border-2 border-dashed transition-all duration-500
                        min-h-[450px] flex flex-col items-center justify-center p-10 text-center cursor-pointer
                        ${isDragging
                            ? 'border-purple-400 bg-purple-950/30 scale-[1.02]'
                            : 'border-slate-700/50 hover:border-purple-500/60 bg-slate-900/30 hover:bg-slate-900/50'}
                    `}
                    onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)}
                    onDrop={handleDrop}
                    onClick={() => fileInputRef.current?.click()}
                >
                    <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

                    <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        accept="video/*"
                        onChange={(e) => e.target.files?.[0] && processFile(e.target.files[0])}
                    />

                    <div className="relative z-10 flex flex-col items-center">
                        <div className="bg-slate-800/80 p-5 rounded-2xl mb-6 group-hover:scale-110 group-hover:shadow-[0_0_40px_rgba(168,85,247,0.2)] transition-all duration-500 border border-slate-700/50">
                            <Video className="w-10 h-10 text-purple-400" />
                        </div>

                        <h2 className="text-2xl md:text-3xl font-bold text-white mb-3 tracking-wide">
                            DROP VIDEO FOR ANALYSIS
                        </h2>
                        <p className="text-slate-400 text-base max-w-md mb-6">
                            Upload a video file and our AI will detect license plates and traffic signs in every frame
                        </p>

                        <div className="flex gap-3 text-xs text-slate-500 font-mono border border-slate-700/50 rounded-full px-5 py-2 bg-slate-900/70">
                            <span>.MP4</span>
                            <span className="text-slate-700">•</span>
                            <span>.AVI</span>
                            <span className="text-slate-700">•</span>
                            <span>.MOV</span>
                            <span className="text-slate-700">•</span>
                            <span>.WEBM</span>
                        </div>
                    </div>
                </div>
            ) : (
                // Processing / Result View
                <div className="space-y-6">
                    {/* Video Preview */}
                    <div className="relative glass-card rounded-2xl overflow-hidden">
                        <video
                            src={result?.download_url ? `http://localhost:8000${result.download_url}` : video}
                            className="w-full max-h-[500px] object-contain bg-black"
                            controls={!isProcessing && result}
                            muted
                        />

                        {/* Processing Overlay */}
                        {isProcessing && (
                            <div className="absolute inset-0 bg-slate-950/80 backdrop-blur-sm flex flex-col items-center justify-center gap-4">
                                <div className="animate-processing">
                                    <FileVideo className="w-16 h-16 text-purple-400" />
                                </div>
                                <p className="text-purple-400 font-semibold tracking-widest text-sm uppercase">
                                    Processing Video...
                                </p>
                                <div className="w-64 h-2 bg-slate-800 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-gradient-to-r from-purple-500 to-cyan-500 transition-all duration-300"
                                        style={{ width: `${progress}%` }}
                                    />
                                </div>
                                <p className="text-slate-500 text-sm">{Math.round(progress)}% Complete</p>
                            </div>
                        )}

                        {/* Close Button */}
                        {!isProcessing && (
                            <button
                                onClick={reset}
                                className="absolute top-3 right-3 bg-slate-900/80 hover:bg-red-500/80 text-white p-2 rounded-lg backdrop-blur border border-slate-700/50 shadow-lg transition-colors"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        )}
                    </div>

                    {/* Error */}
                    {error && (
                        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 flex items-center gap-3">
                            <AlertTriangle className="w-5 h-5" />
                            <p>{error}</p>
                        </div>
                    )}

                    {/* Action Buttons */}
                    {!result && !isProcessing && (
                        <div className="flex justify-center gap-4">
                            <button
                                onClick={startProcessing}
                                className="px-8 py-3 bg-gradient-to-r from-purple-500 to-cyan-500 hover:from-purple-400 hover:to-cyan-400 text-white font-bold rounded-xl transition-all shadow-lg shadow-purple-500/30 flex items-center gap-2"
                            >
                                <Play className="w-5 h-5" />
                                Start Processing
                            </button>
                            <button
                                onClick={reset}
                                className="px-6 py-3 bg-slate-800 hover:bg-slate-700 text-white rounded-xl transition-all border border-slate-700"
                            >
                                Cancel
                            </button>
                        </div>
                    )}

                    {/* Results */}
                    {result && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            {/* Stats Card */}
                            <div className="glass-panel rounded-xl p-6">
                                <h3 className="text-lg font-bold text-white mb-4 flex items-center gap-2">
                                    <FileVideo className="w-5 h-5 text-purple-400" />
                                    Processing Complete
                                </h3>
                                <div className="space-y-3 text-sm">
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Frames Processed</span>
                                        <span className="text-white font-mono">{result.stats?.processed_frames || 0}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Plates Detected</span>
                                        <span className="text-cyan-400 font-mono">{result.stats?.plates_detected || 0}</span>
                                    </div>
                                    <div className="flex justify-between">
                                        <span className="text-slate-400">Signs Detected</span>
                                        <span className="text-orange-400 font-mono">{result.stats?.signs_detected || 0}</span>
                                    </div>
                                </div>

                                <button
                                    onClick={downloadVideo}
                                    className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-emerald-500 to-cyan-500 hover:from-emerald-400 hover:to-cyan-400 text-white font-bold rounded-xl transition-all shadow-lg flex items-center justify-center gap-2"
                                >
                                    <Download className="w-5 h-5" />
                                    Download Processed Video
                                </button>
                            </div>

                            {/* Detections Summary */}
                            <div className="glass-panel rounded-xl p-6">
                                <h3 className="text-lg font-bold text-white mb-4">Detected Items</h3>

                                {result.stats?.unique_plates?.length > 0 && (
                                    <div className="mb-4">
                                        <h4 className="text-sm text-cyan-400 font-semibold mb-2 flex items-center gap-2">
                                            <Car className="w-4 h-4" />
                                            License Plates
                                        </h4>
                                        <div className="flex flex-wrap gap-2">
                                            {result.stats.unique_plates.map((plate, i) => (
                                                <span key={i} className="px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400 font-mono text-sm">
                                                    {plate}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {result.stats?.unique_signs?.length > 0 && (
                                    <div>
                                        <h4 className="text-sm text-orange-400 font-semibold mb-2 flex items-center gap-2">
                                            <AlertTriangle className="w-4 h-4" />
                                            Traffic Signs
                                        </h4>
                                        <div className="flex flex-wrap gap-2">
                                            {result.stats.unique_signs.map((sign, i) => (
                                                <span key={i} className="px-3 py-1 bg-orange-500/10 border border-orange-500/30 rounded-lg text-orange-400 text-sm">
                                                    {sign}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {!result.stats?.unique_plates?.length && !result.stats?.unique_signs?.length && (
                                    <p className="text-slate-500 text-center py-4">No detections found in video</p>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default VideoMode;
