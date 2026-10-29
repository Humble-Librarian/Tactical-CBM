/**
 * Tactical-CBM Engine - WebAssembly Geometry Loader & Dual-Engine Parity Fallback
 * Compiles and exposes canonical C++ geometric functions from tactical_geometry.wasm.
 */

let wasmGeometry = null;

function initWasmGeometry() {
    if (typeof window !== 'undefined' && window.WebAssembly) {
        return fetch('tactical_geometry.wasm')
            .then(res => {
                if (res.ok) return res.arrayBuffer();
                throw new Error("WASM binary not found");
            })
            .then(bytes => WebAssembly.instantiate(bytes))
            .then(res => {
                wasmGeometry = res.instance.exports;
                console.log("[WASM Engine] tactical_geometry.wasm loaded successfully! C++ parity active.");
                return wasmGeometry;
            })
            .catch(err => {
                console.log("[WASM Engine] Using V8 JIT inline math fallback:", err.message);
                return null;
            });
    }
    return Promise.resolve(null);
}

// Immediate bootstrap
initWasmGeometry();
