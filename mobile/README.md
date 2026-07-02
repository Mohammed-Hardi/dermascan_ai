# DermaScan AI Mobile

React Native client for the DermaScan AI FastAPI backend. This starter uses
Expo and keeps the current Streamlit app untouched.

## Setup

From the repository root:

```powershell
cd mobile
npm install
```

## Run

Start the FastAPI backend from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --port 8000
```

Then start the mobile app:

```powershell
cd mobile
npm run start
```

Use the Expo app, an emulator, or the web target.

## API URL

The default API URL is configured in `app.json` as:

```json
"apiBaseUrl": "http://127.0.0.1:8000"
```

Use your computer LAN IP when testing on a physical phone, for example
`http://192.168.1.20:8000`. Android emulators often need
`http://10.0.2.2:8000`.

## Current Flow

- Check backend availability with `GET /health`
- Pick an image from the device library
- Submit metadata and the image to `POST /api/predict`
- Render cautious, non-diagnostic results returned by the API

The app is for education and development only. It must not be used for real
screening or diagnosis.
