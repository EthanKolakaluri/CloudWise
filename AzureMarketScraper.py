import aiohttp
import asyncio
import json
from azure.identity import DefaultAzureCredential

# Initialize the credential object
credential = DefaultAzureCredential()

# Get the access token for authentication
access_token = credential.get_token("https://management.azure.com/.default").token

headers = {
    "Authorization": f"Bearer {access_token}"
}

# Define your API URLs (adjust these with actual endpoints)
vm_url = "https://management.azure.com/subscriptions/{subscription_id}/providers/Microsoft.Compute/virtualMachines?api-version=2021-03-01"
pricing_url = "https://prices.azure.com/api/retail/prices?api-version=2021-01-01"
metrics_url = "https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.Insights/metrics?api-version=2018-01-01"

# Async function to fetch data from a single API
async def fetch_data(session, url):
    async with session.get(url, headers={"Authorization": f"Bearer {access_token}"}) as response:
        data = await response.json()
        return data

# Function to fetch all APIs concurrently
async def fetch_all_data():
    async with aiohttp.ClientSession() as session:
        # Initiate all fetch tasks concurrently
        vm_data = fetch_data(session, vm_url)
        pricing_data = fetch_data(session, pricing_url)
        metrics_data = fetch_data(session, metrics_url)
        
        # Wait until all tasks are completed
        results = await asyncio.gather(vm_data, pricing_data, metrics_data)
        
        # Consolidate the results into one dataset
        dataset = {
            "vm_data": results[0],
            "pricing_data": results[1],
            "metrics_data": results[2]
        }
        
        return dataset

# Run the async function
async def main():
    data = await fetch_all_data()
    
    # Print out or save the resulting data
    with open("AzureData.json", "w") as json_file:
        json.dump(data, json_file, indent=4)
    
    print("Data has been saved to azure_data.json")

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
