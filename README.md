# ARGUS — Multi-Camera Re-Identification

A practical, privacy-conscious multi-camera person re-identification system for authorized environments.

ARGUS detects and tracks people inside each camera stream, extracts appearance embeddings, links tracklets across cameras into temporary global identities, stores events in SQLite, and exposes both a CLI and a Streamlit dashboard.

> **Scope:** This project is designed for consented test footage, labs, research, retail/industrial analytics, sports, and other authorized deployments. It does **not** identify real-world names, perform face recognition, or connect to external identity databases.

## Features

- Person detection and per-camera tracking
- Multi-camera appearance re-identification
- Track-level embedding aggregation for more stable matches
- Temporal/topology constraints between cameras
- Temporary global IDs rather than real-world identity
- Searchable event timeline
- SQLite persistence
- Annotated video export
- Streamlit dashboard
- CLI batch processing
- Synthetic unit tests for identity linking
- Docker and GitHub Actions CI

## Architecture

```
Camera / Video
     |
     v
Detector + ByteTrack
     |
     v
Per-camera tracklets
     |
     +----> appearance crops ----> embedding model
                                  |
                                  v
                         GlobalIdentityManager
                         /        |        \
                similarity   topology   timing
                         \        |        /
                                  v
                            Global ID
                                  |
                         SQLite event store
                                  |
                         Dashboard / CLI
```

## Quick start

Python 3.11 is recommended.

```bash
git clone https://github.com/ahpa5246-lgtm/Multi-Camera-Re-Identification.git
cd Multi-Camera-Re-Identification

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first real inference run downloads pretrained model weights. A GPU is optional; CUDA is used automatically when available.

## CLI

```bash
python -m mcreid.cli \
  --input cam01=videos/entrance.mp4 \
  --input cam02=videos/hall.mp4 \
  --output runs/demo
```

Optional camera topology:

```bash
python -m mcreid.cli \
  --input cam01=videos/a.mp4 \
  --input cam02=videos/b.mp4 \
  --topology config/topology.example.yaml \
  --output runs/demo
```

## Camera topology

`config/topology.example.yaml` describes plausible transitions:

```yaml
transitions:
  cam01:
    cam02: {min_seconds: 0, max_seconds: 180}
```

A candidate global identity is rejected when a transition violates a configured travel-time window. Unknown camera pairs are still allowed unless `strict_topology` is enabled.

## Dashboard workflow

1. Upload two or more videos.
2. Give each camera a stable name.
3. Start processing.
4. Review the camera outputs, global identities, and event timeline.
5. Select a global ID to inspect only that track across cameras.

Uploaded video files are processed locally by the machine running Streamlit.

## Models

The default detector/tracker uses Ultralytics YOLO with ByteTrack. Appearance embeddings use a pretrained `timm` visual backbone. Both can be replaced behind the small interfaces in `mcreid/detection.py` and `mcreid/embedding.py`.

The project intentionally does not include face recognition.

### Ultralytics licensing

Ultralytics packages/models have their own licensing terms. Review those terms before commercial deployment. The ARGUS source code in this repository is MIT licensed.

## Configuration

Core settings are defined by `PipelineConfig` in `mcreid/config.py`. Important values include:

- `detector_model`
- `detection_confidence`
- `reid_similarity_threshold`
- `min_crop_height`
- `embedding_history`
- `max_idle_seconds`
- `device`

The defaults favor a working demo over maximum throughput.

## Privacy and deployment notes

For real deployments:

- obtain authorization for all camera feeds;
- disclose monitoring where required;
- define retention periods;
- restrict access to raw footage and embeddings;
- encrypt stored data;
- avoid using global IDs as claims of legal identity;
- validate false-match rates on your own cameras;
- add human review for consequential decisions.

Appearance re-identification is probabilistic. Clothing changes, occlusion, poor lighting, similar uniforms, and long gaps can cause incorrect matches.

## Tests

```bash
pytest -q
```

The tests do not download model weights; they focus on matching, topology constraints, configuration, and event persistence.

## Project layout

```
.
├── app.py
├── mcreid/
│   ├── cli.py
│   ├── config.py
│   ├── detection.py
│   ├── embedding.py
│   ├── identity.py
│   ├── pipeline.py
│   ├── storage.py
│   ├── topology.py
│   └── visualization.py
├── config/
│   └── topology.example.yaml
├── tests/
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

## License

MIT. Model weights and third-party packages retain their own licenses.
