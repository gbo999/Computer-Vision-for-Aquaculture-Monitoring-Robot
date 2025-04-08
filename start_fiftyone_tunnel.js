const { spawn } = require('child_process');
const localtunnel = require('localtunnel');

// FiftyOne app port
const PORT = 5180; // Change this to 5186 if needed

async function startTunnel() {
  try {
    // Create the tunnel
    const tunnel = await localtunnel({ 
      port: PORT,
      subdomain: 'fiftyone-app-' + Math.random().toString(36).substring(2, 8) // Unique subdomain
    });
    
    console.log('Tunnel URL:', tunnel.url);
    
    // Handle tunnel events
    tunnel.on('close', () => {
      console.log('Tunnel closed');
      process.exit(1);
    });
    
    tunnel.on('error', (err) => {
      console.error('Tunnel error:', err);
      process.exit(1);
    });
    
    // Keep the script running
    process.on('SIGINT', () => {
      console.log('Closing tunnel...');
      tunnel.close();
      process.exit(0);
    });
    
    console.log('Tunnel is running. Press Ctrl+C to stop.');
  } catch (error) {
    console.error('Failed to start tunnel:', error);
    process.exit(1);
  }
}

startTunnel(); 