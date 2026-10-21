"""Start the scoring service.

    python -m src.app.main                      # http://127.0.0.1:8000  (API docs at /docs, form at /ui)
    python -m src.app.main --host 0.0.0.0 --port 8080
"""
import argparse

import uvicorn


def main(argv=None):
    parser = argparse.ArgumentParser(description='Start the Home Credit scoring service.')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--reload', action='store_true', help='restart on code changes (development)')
    args = parser.parse_args(argv)
    uvicorn.run('src.app.app:app', host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()
