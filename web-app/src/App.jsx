import React, { useState } from 'react';
import { Camera, Upload, Video, Car, ShieldCheck, Zap, AlertTriangle } from 'lucide-react';
import UploadMode from './components/UploadMode';
import CameraMode from './components/CameraMode';
import VideoMode from './components/VideoMode';

function App() {
    const [activeMode, setActiveMode] = useState('welcome'); // welcome, upload, live, video

    return (
        <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-cyan-500 selection:text-white pb-20">
            {/* Navigation */}
            <nav className="fixed top-0 w-full z-50 bg-slate-900/80 backdrop-blur-md border-b border-slate-700/50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">
                        <div className="flex items-center gap-2 cursor-pointer" onClick={() => setActiveMode('welcome')}>
                            <Car className="w-8 h-8 text-cyan-400" />
                            <span className="text-xl font-bold tracking-tight text-white">
                                Vision<span className="text-cyan-400">AI</span> Pro
                            </span>
                        </div>

                        <div className="flex items-center gap-2 md:gap-4">
                            <button
                                onClick={() => setActiveMode('upload')}
                                className={`px-3 md:px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2
                  ${activeMode === 'upload'
                                        ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-500/20'
                                        : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
                            >
                                <Upload className="w-4 h-4" />
                                <span className="hidden sm:inline">Image</span>
                            </button>
                            <button
                                onClick={() => setActiveMode('video')}
                                className={`px-3 md:px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2
                  ${activeMode === 'video'
                                        ? 'bg-purple-500 text-white shadow-lg shadow-purple-500/20'
                                        : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
                            >
                                <Video className="w-4 h-4" />
                                <span className="hidden sm:inline">Video</span>
                            </button>
                            <button
                                onClick={() => setActiveMode('live')}
                                className={`px-3 md:px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 flex items-center gap-2
                  ${activeMode === 'live'
                                        ? 'bg-rose-500 text-white shadow-lg shadow-rose-500/20'
                                        : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}
                            >
                                <Camera className="w-4 h-4" />
                                <span className="hidden sm:inline">Live</span>
                            </button>
                        </div>
                    </div>
                </div>
            </nav>

            {/* Main Content Area */}
            <main className="pt-24 px-4 max-w-7xl mx-auto min-h-[calc(100vh-100px)] flex flex-col">

                {activeMode === 'welcome' && (
                    <div className="flex-1 flex flex-col items-center justify-center text-center animate-in fade-in slide-in-from-bottom-8 duration-700">
                        <div className="inline-flex items-center justify-center p-2 mb-8 rounded-full bg-slate-800/50 border border-slate-700 backdrop-blur-sm">
                            <span className="px-3 py-1 text-xs font-semibold text-cyan-400 uppercase tracking-wider">
                                AI System Online
                            </span>
                        </div>

                        <h1 className="text-5xl md:text-7xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-white via-slate-200 to-slate-400 mb-6 tracking-tight">
                            VisionAI Pro
                        </h1>

                        <p className="max-w-2xl text-lg md:text-xl text-slate-400 mb-10 leading-relaxed">
                            Advanced AI-powered detection for <span className="text-cyan-400 font-semibold">License Plates</span> and{' '}
                            <span className="text-orange-400 font-semibold">Traffic Signs</span>.
                            <br className="hidden md:block" />
                            Real-time analysis for images, videos, and live camera feeds.
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4 w-full sm:w-auto">
                            <button
                                onClick={() => setActiveMode('upload')}
                                className="group relative px-8 py-4 bg-cyan-500 hover:bg-cyan-400 text-white rounded-xl font-bold text-lg transition-all duration-300 shadow-xl shadow-cyan-500/20 hover:shadow-cyan-500/40 hover:-translate-y-1"
                            >
                                <div className="absolute inset-0 rounded-xl bg-white/20 opacity-0 group-hover:opacity-100 transition-opacity" />
                                <span className="flex items-center justify-center gap-3">
                                    <Upload className="w-5 h-5" />
                                    Upload Image
                                </span>
                            </button>

                            <button
                                onClick={() => setActiveMode('video')}
                                className="group px-8 py-4 bg-purple-600 hover:bg-purple-500 text-white rounded-xl font-bold text-lg transition-all duration-300 shadow-xl shadow-purple-500/20 hover:shadow-purple-500/40 hover:-translate-y-1"
                            >
                                <span className="flex items-center justify-center gap-3">
                                    <Video className="w-5 h-5" />
                                    Process Video
                                </span>
                            </button>

                            <button
                                onClick={() => setActiveMode('live')}
                                className="group px-8 py-4 bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 hover:border-slate-600 rounded-xl font-bold text-lg transition-all duration-300 hover:-translate-y-1"
                            >
                                <span className="flex items-center justify-center gap-3">
                                    <Camera className="w-5 h-5 text-rose-500" />
                                    Live Camera
                                </span>
                            </button>
                        </div>

                        <div className="mt-20 grid grid-cols-1 md:grid-cols-4 gap-6 text-left w-full max-w-5xl opacity-80">
                            <div className="p-6 rounded-2xl bg-slate-800/30 border border-slate-700/50 backdrop-blur-sm">
                                <Car className="w-8 h-8 text-cyan-400 mb-4" />
                                <h3 className="text-lg font-bold text-white mb-2">Plate Detection</h3>
                                <p className="text-slate-400 text-sm">High-accuracy license plate recognition with OCR text extraction.</p>
                            </div>
                            <div className="p-6 rounded-2xl bg-slate-800/30 border border-slate-700/50 backdrop-blur-sm">
                                <AlertTriangle className="w-8 h-8 text-orange-400 mb-4" />
                                <h3 className="text-lg font-bold text-white mb-2">Traffic Signs</h3>
                                <p className="text-slate-400 text-sm">Detect and classify traffic signs including Stop, Speed Limit, and more.</p>
                            </div>
                            <div className="p-6 rounded-2xl bg-slate-800/30 border border-slate-700/50 backdrop-blur-sm">
                                <Zap className="w-8 h-8 text-amber-400 mb-4" />
                                <h3 className="text-lg font-bold text-white mb-2">Real-time</h3>
                                <p className="text-slate-400 text-sm">Instant frame-by-frame analysis with live camera support.</p>
                            </div>
                            <div className="p-6 rounded-2xl bg-slate-800/30 border border-slate-700/50 backdrop-blur-sm">
                                <ShieldCheck className="w-8 h-8 text-emerald-400 mb-4" />
                                <h3 className="text-lg font-bold text-white mb-2">Video Processing</h3>
                                <p className="text-slate-400 text-sm">Upload videos and download annotated output with all detections.</p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Components */}
                {activeMode === 'upload' && <UploadMode />}
                {activeMode === 'live' && <CameraMode />}
                {activeMode === 'video' && <VideoMode />}

            </main>

            <footer className="fixed bottom-0 w-full py-4 text-center text-slate-600 text-xs bg-slate-900/90 backdrop-blur border-t border-slate-800/50 z-40">
                <p>© 2024 VisionAI Pro. Advanced AI Computer Vision System.</p>
            </footer>
        </div>
    )
}

export default App
