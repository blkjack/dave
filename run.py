import os
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run the Advanced Data Analyzer app')
    parser.add_argument('--env', choices=['dev', 'prod'], default='dev',
                      help='Environment to run (dev or prod)')
    args = parser.parse_args()

    # Set environment variable
    os.environ['ENVIRONMENT'] = args.env

    # Determine which app to run
    if args.env == 'dev':
        app_path = 'src/dev/app.py'
    else:
        app_path = 'src/prod/app.py'

    # Run the appropriate app
    os.system(f'streamlit run {app_path}')

if __name__ == '__main__':
    main() 