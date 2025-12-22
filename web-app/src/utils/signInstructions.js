// Road sign to user-readable instruction mapping
// Maps traffic sign class names (from backend) to meaningful driver instructions

export const SIGN_INSTRUCTIONS = {
    // These match the TRAFFIC_SIGN_CLASSES in app.py (lines 57-73)
    "Speed Limit": "🚗 Speed Limit - Observe the posted speed limit and adjust your speed accordingly",
    "Stop": "🛑 STOP - Come to a complete stop before proceeding",
    "Yield": "🔺 Yield - Slow down and give way to other traffic",
    "No Entry": "⛔ No Entry - Entry is prohibited in this direction",
    "Pedestrian Crossing": "🚶 Pedestrian Crossing - Slow down and watch for pedestrians",
    "Traffic Light": "🚦 Traffic Light - Be prepared to stop at the signal",
    "Turn Right": "➡️ Turn Right - Right turn ahead or required",
    "Turn Left": "⬅️ Turn Left - Left turn ahead or required",
    "Straight Only": "⬆️ Straight Only - Continue straight, no turns allowed",
    "No Parking": "🅿️ No Parking - Parking is prohibited in this zone",
    "Warning": "⚠️ Warning - Caution: hazard ahead, proceed carefully",
    "School Zone": "🏫 School Zone - Reduce speed, watch for children crossing",
    "Hospital": "🏥 Hospital Zone - Keep quiet, emergency vehicles may be present",
    "Railway Crossing": "🚂 Railway Crossing - Stop, look, and listen for trains",
    "One Way": "➡️ One Way - Traffic flows in one direction only",

    // Additional signs that might be detected (using underscore format from YAML)
    bus_stop: "🚌 Bus Stop Ahead - Expect buses stopping and passengers crossing",
    do_not_enter: "⛔ Do Not Enter - Entry is prohibited in this direction",
    do_not_stop: "🚫 No Stopping - Stopping is not allowed in this area",
    do_not_turn_l: "↰ No Left Turn - Left turn is prohibited here",
    do_not_turn_r: "↱ No Right Turn - Right turn is prohibited here",
    do_not_u_turn: "⤾ No U-Turn - U-turn is not allowed at this location",
    no_parking: "🅿️ No Parking - Parking is prohibited in this zone",
    enter_left_lane: "⬅️ Use Left Lane - Move to the left lane ahead",
    left_right_lane: "↔️ Choose Lane - Select left or right lane as needed",
    parking: "🅿️ Parking Available - Parking is permitted here",
    ped_crossing: "🚶 Pedestrian Crossing Ahead - Slow down and watch for pedestrians",
    ped_zebra_cross: "🦓 Zebra Crossing - Yield to pedestrians on crosswalk",
    railway_crossing: "🚂 Railway Crossing Ahead - Stop, look, and listen for trains",
    green_light: "🟢 Green Light - Proceed with caution",
    yellow_light: "🟡 Yellow Light - Prepare to stop, proceed only if safe",
    red_light: "🔴 Red Light - Stop completely and wait",
    traffic_light: "🚦 Traffic Light Ahead - Be prepared to stop",
    stop: "🛑 STOP - Come to a complete stop before proceeding",
    u_turn: "↩️ U-Turn Permitted - You may make a U-turn here",
    t_intersection_l: "⊥ T-Intersection Ahead - Prepare to turn left or right",
    warning: "⚠️ Warning - Caution: hazard ahead, proceed carefully",
    school_zone: "🏫 School Zone - Reduce speed, watch for children crossing"
};

/**
 * Get user-readable instruction for a road sign
 * @param {string} className - The class name from YOLO detection
 * @returns {object} - Object with instruction text and formatted display name
 */
export const getSignInstruction = (className) => {
    if (!className) {
        return {
            instruction: "Road sign detected",
            displayName: "Unknown Sign",
            hasInstruction: false
        };
    }

    // Try exact match first
    let instruction = SIGN_INSTRUCTIONS[className];

    // If not found, try lowercase with underscores
    if (!instruction) {
        const normalizedKey = className.toLowerCase().replace(/\s+/g, '_');
        instruction = SIGN_INSTRUCTIONS[normalizedKey];
    }

    if (instruction) {
        return {
            instruction: instruction,
            displayName: className,
            hasInstruction: true
        };
    }

    return {
        instruction: `⚠️ ${className} detected - Please observe this road sign`,
        displayName: className,
        hasInstruction: false
    };
};

/**
 * Format class name to human-readable display name
 * @param {string} className - The class name from YOLO detection
 * @returns {string} - Formatted display name
 */
export const formatClassName = (className) => {
    if (!className) return 'Unknown Sign';

    // If already has spaces, return as-is
    if (className.includes(' ')) {
        return className;
    }

    // Convert underscores to spaces and capitalize
    return className
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
};
