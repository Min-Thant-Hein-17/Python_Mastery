# Disaster Management Practice Assignment

## Disaster Relief Resource Management
You are tasked with developing a simple system to manage disaster relief efforts following a natural disaster. Your system will help track different types of relief resources being distributed, the regions they are sent to, and the status of the affected regions.

### Problem Requirements:

*   **List of Affected Regions:** 
    *   Initialize a list of strings representing regions affected by the disaster.
    *   Each region is a string.
*   **Resource Inventory:** 
    *   Initialize a dictionary where the keys are resource names (strings) and the values are the available quantities (integers) of each resource.
*   **Resource Distribution Tracking:** 
    *   Create an empty dictionary to record the distribution of resources to each region.
    *   Distribute resources to regions by performing the following steps manually in your code:
        1.  **Select a Region:** Choose a region from the `affected_regions` list.
        2.  **Select a Resource:** Choose a resource from the `resource_inventory` dictionary.
        3.  **Specify Quantity:** Determine the quantity of the resource to distribute.
        4.  **Check Availability:** Verify that the selected resource exists in `resource_inventory`. Ensure that the desired quantity does not exceed the available quantity in `resource_inventory`.
        5.  **Update Inventory and Distribution:** Deduct the distributed quantity from the `resource_inventory`. Update the `resource_distribution` dictionary to reflect the resources sent to the selected region.
*   **Tracking Region Status:** 
    *   Initialize a dictionary to store the current status of each region.
    *   The status can be "Critical", "Stable", or "Recovered".
    *   Update the status of any region by modifying the `region_status` dictionary as needed.
*   **Report Generation:** 
    *   After performing several distribution and status update operations, generate a summary report that includes:
        *   **Resources Distributed:** List each region along with the resources and quantities distributed to it.
        *   **Region Status:** Display the current status of each region.
        *   **Remaining Inventory:** Show the remaining quantities of each resource in `resource_inventory`.

---

### Additional Exercises
To further enhance your skills, consider performing the following additional tasks:

*   **Adding New Regions:** Add a new region to the `affected_regions` list and initialize its status in `region_status`.
*   **Restocking Resources:** Increase the quantity of a specific resource in `resource_inventory`.
*   **Removing Regions:** Remove a region from `affected_regions` and delete its entries from `region_status` and `resource_distribution`.
*   **Advanced Reporting:** 
    *   Calculate the total number of each resource distributed across all regions.
    *   Identify regions that have received the most resources.
*   **Input Handling:** Allow user input to dynamically distribute resources, update statuses, and generate reports.
