import json, os, sys
from dotenv import load_dotenv
load_dotenv()
sys.path.insert(0, '.')

with open('config.json') as f:
    cfg = json.load(f)

email     = os.getenv('JIRA_EMAIL') or cfg['jira']['email']
api_token = os.getenv('JIRA_API_TOKEN') or cfg['jira']['api_token']

print('Email:', email)
print('Token length:', len(api_token))

from jira import JIRA
print('Connecting to Jira...')
try:
    jira = JIRA(server=cfg['jira']['url'], basic_auth=(email, api_token))
    print('Connected:', jira.server_url)
except Exception as e:
    print('CONNECT ERROR:', e)
    sys.exit(1)

jql = 'project = SAC AND status in (Resolved, Cancelled, Closed) AND "Reporting Area[Dropdown]" = "Order Fallout" AND updated >= -10d ORDER BY updated DESC'
print('\nRunning JQL:', jql)
try:
    issues = jira.search_issues(jql, maxResults=5, fields='summary,status')
    print(f'Found tickets: {len(issues)}')
    for i in issues:
        print(f'  {i.key}: {i.fields.summary[:70]}')
except Exception as e:
    print('JQL ERROR:', e)
