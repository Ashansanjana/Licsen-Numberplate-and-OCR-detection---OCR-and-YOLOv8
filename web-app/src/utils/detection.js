// Backend API integration for combined detection (plates + traffic signs)

export const detectAll = async (width, height, imageFile) => {
    console.log("🔍 detectAll called with:", { width, height, imageFile });

    if (!imageFile) {
        console.warn("❌ No file provided for detection");
        return { plates: [], traffic_signs: [] };
    }

    console.log("📤 Sending request to backend...");

    try {
        const formData = new FormData();
        formData.append('file', imageFile);

        const response = await fetch('http://localhost:8000/detect', {
            method: 'POST',
            body: formData,
        });

        console.log("📥 Response received:", response.status, response.statusText);

        if (!response.ok) {
            throw new Error(`API Error: ${response.statusText}`);
        }

        const data = await response.json();
        console.log("✅ Detection result:", data);

        return {
            plates: data.plates || data.detections || [],
            traffic_signs: data.traffic_signs || []
        };

    } catch (error) {
        console.error("❌ Detection failed:", error);
        return { plates: [], traffic_signs: [] };
    }
};

// Backward compatibility - keep simulateDetection for CameraMode
export const simulateDetection = async (width, height, imageFile) => {
    const result = await detectAll(width, height, imageFile);
    return result.plates || [];
};

// Video processing API
export const processVideo = async (videoFile, onProgress) => {
    console.log("🎬 processVideo called");

    if (!videoFile) {
        console.warn("❌ No video file provided");
        return null;
    }

    try {
        const formData = new FormData();
        formData.append('file', videoFile);

        const response = await fetch('http://localhost:8000/detect/video', {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Video processing failed: ${response.statusText}`);
        }

        const data = await response.json();
        console.log("✅ Video processing complete:", data);
        return data;

    } catch (error) {
        console.error("❌ Video processing failed:", error);
        throw error;
    }
};
