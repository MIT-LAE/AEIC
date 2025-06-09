import requests
from bs4 import BeautifulSoup
import csv

# URL of the ECMWF L137 model level definitions page
url = 'https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions'

# Send a GET request to the URL
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing the coefficients
    # This assumes that the table is the first one on the page
    table = soup.find('table')

    # Initialize lists to store the coefficients
    levels = []
    a_coeffs = []
    b_coeffs = []

    # Iterate over the rows of the table
    for row in table.find_all('tr')[1:]:  # Skip the header row
        cols = row.find_all('td')
        if len(cols) >= 3:
            level = cols[0].get_text(strip=True)
            a = cols[1].get_text(strip=True)
            b = cols[2].get_text(strip=True)
            levels.append(level)
            a_coeffs.append(a)
            b_coeffs.append(b)

    # Write the coefficients to a CSV file
    with open('L137_coefficients.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['level', 'a', 'b'])
        for level, a, b in zip(levels, a_coeffs, b_coeffs):
            writer.writerow([level, a, b])

    print('Coefficients have been written to L137_coefficients.csv')
else:
    print(f'Failed to retrieve the page. Status code: {response.status_code}')
