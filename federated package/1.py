from diagrams import Diagram, Cluster
from diagrams.generic.os import Windows
from diagrams.onprem.network import Internet
from diagrams.programming.framework import AngularJS
from diagrams.onprem.compute import Server

# Initialize the diagram
with Diagram("Safety-Focused Wireless Deceleration/Brake System Architecture", show=False, direction="TB") as diag:
    
    # Define components
    smartphone_app = AngularJS("Smartphone App")
    wireless_module = Server("Wireless Module (Wi-Fi)")
    primary_mcu = Server("MCU (Primary)")
    secondary_mcu = Server("MCU (Secondary)")
    electromagnetic_brake = Server("Electromagnetic Brake Assembly")
    battery_backup1 = Server("Battery Backup")
    battery_backup2 = Server("Battery Backup")
    
    # Connect components to represent architecture
    smartphone_app >> wireless_module >> primary_mcu
    primary_mcu >> electromagnetic_brake
    primary_mcu >> secondary_mcu
    primary_mcu >> battery_backup1
    secondary_mcu >> battery_backup2
    diag