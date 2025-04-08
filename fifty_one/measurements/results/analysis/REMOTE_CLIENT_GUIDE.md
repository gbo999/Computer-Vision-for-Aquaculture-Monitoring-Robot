# FiftyOne Remote Client Connection Guide

This guide will help you connect to a remote FiftyOne server to view the prawn measurement datasets.

## Prerequisites

1. **Python Installation**: You need Python 3.6 or later installed on your computer.
   - Download from [python.org](https://www.python.org/downloads/)

2. **FiftyOne Installation**: Install the FiftyOne package on your computer:
   ```
   pip install fiftyone
   ```

## Connecting to the Remote Server

The person hosting the dataset will provide you with connection details that look like this:

```
fiftyone app connect --destination 192.168.1.123:5252 --port 5151
```

Where:
- `192.168.1.123` is the server's IP address
- `5252` is the remote port
- `5151` is the local port for viewing the app

### Step 1: Open Your Terminal or Command Prompt

- **Windows**: Search for "cmd" or "PowerShell" in the start menu
- **Mac**: Open Terminal from Applications > Utilities
- **Linux**: Open your terminal application

### Step 2: Run the Connection Command

Copy and paste the connection command provided by the server administrator:

```
fiftyone app connect --destination <SERVER_IP>:<REMOTE_PORT> --port <LOCAL_PORT>
```

For example:
```
fiftyone app connect --destination 192.168.1.123:5252 --port 5151
```

### Step 3: View the Dataset

After running the command:

1. You should see a message that says "Connected to remote session"
2. A web browser should automatically open with the FiftyOne interface
3. If a browser doesn't open automatically, manually navigate to: `http://localhost:<LOCAL_PORT>`

## Navigating the FiftyOne Interface

Once connected, you can:

1. Browse through the dataset samples in the main panel
2. Use the sidebar to filter samples
3. Click on individual samples to view details
4. Use the visualization tools to analyze the keypoint detections

## Troubleshooting

If you encounter issues:

1. **Connection Refused**: Make sure the server is running and check if there's a firewall blocking the connection.

2. **Cannot Find FiftyOne Command**: Make sure you've installed FiftyOne correctly.

3. **Browser Doesn't Open**: Try manually navigating to `http://localhost:<LOCAL_PORT>` in your web browser.

4. **Other Issues**: Contact the server administrator for assistance.

## Disconnecting

To close the connection, simply press `Ctrl+C` in the terminal/command prompt where the FiftyOne client is running.

---

## For Advanced Users: Connection Options

You can customize your connection with additional options:

```
fiftyone app connect --destination <SERVER_IP>:<REMOTE_PORT> --port <LOCAL_PORT> --desktop
```

The `--desktop` flag opens the FiftyOne desktop app instead of the web browser if you have it installed. 