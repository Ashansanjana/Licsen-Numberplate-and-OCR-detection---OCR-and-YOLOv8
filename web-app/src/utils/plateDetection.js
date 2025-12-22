// Backend API integration for plate detection

export const simulateDetection = async (width, height, imageFile) => {
    console.log("🔍 simulateDetection called with:", { width, height, imageFile });

    // If no file provided, return empty
    if (!imageFile) {
        console.warn("❌ No file provided for detection");
        return [];
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
        return data.detections || [];

    } catch (error) {
        console.error("❌ Detection failed:", error);
        return [];
    }
};
