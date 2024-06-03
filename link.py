import re
import requests
import time

# Function to extract URLs from text content
def extract_urls(text):
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return urls

# Function to check URL safety using VirusTotal API
def check_url_safety(url, api_key):
    headers = {
        "x-apikey": api_key
    }
    data = {
        'url': url
    }
    response = requests.post('https://www.virustotal.com/api/v3/urls', headers=headers, data=data)
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return None

# Function to get URL analysis report from VirusTotal
def get_url_report(analysis_id, api_key):
    headers = {
        "x-apikey": api_key
    }
    response = requests.get(f'https://www.virustotal.com/api/v3/analyses/{analysis_id}', headers=headers)
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return None

# Example text content
text_content = """
Here are some example URLs: 
1. http://example.com
2. https://www.google.com
3. Visit my website: https://www.mywebsite.com
"""

# Replace 'YOUR_API_KEY' with your actual VirusTotal API key
api_key = 'YOUR API KEY'

# Extract URLs from text content
urls = extract_urls(text_content)

# Check safety of each URL and print the result
for url in urls:
    print(f"Checking safety of URL: {url}")
    submission_result = check_url_safety(url, api_key)
    if submission_result and 'data' in submission_result:
        analysis_id = submission_result['data']['id']
        print(f"Analysis ID for {url}: {analysis_id}")
        
        report = None
        for _ in range(30):  # Retry up to 30 times
            time.sleep(15)  # Wait 15 seconds before each check
            report = get_url_report(analysis_id, api_key)
            if report and 'data' in report and 'attributes' in report['data'] and 'last_analysis_stats' in report['data']['attributes']:
                break
        
        if report and 'data' in report and 'attributes' in report['data'] and 'last_analysis_stats' in report['data']['attributes']:
            attributes = report['data']['attributes']
            total_votes = attributes['last_analysis_stats']
            if total_votes['malicious'] > 0:
                print(f"URL: {url} is unsafe.")
            else:
                print(f"URL: {url} is safe.")
        else:
            print(f"Analysis report for {url} is not complete yet.")
    else:
        print(f"Error occurred while checking URL: {url}")
