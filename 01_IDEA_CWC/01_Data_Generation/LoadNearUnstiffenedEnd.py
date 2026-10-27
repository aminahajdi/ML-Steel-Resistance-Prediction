#import os
import ideastatica_connection_api.connection_api_service_attacher as connection_api_service_attacher
from ideastatica_connection_api import ConLoadEffectMemberLoad, ConLoadEffectPositionEnum, ConLoadEffectSectionLoad, ConLoadEffect
import ideastatica_connection_api
from pathlib import Path
import numpy as np
#import math
#import subprocess
import time
import json
from Functions import read_xlsx, change_cross_section
import random

# Start API service automatically
# Path to IdeaStatiCa.ConnectionRestApi.exe typically in C:\Program Files\IDEA StatiCa\StatiCa 25.0
#get_api_path = r"C:/Program Files/IDEA StatiCa/StatiCa 25.0/IdeaStatiCa.ConnectionRestApi.exe"

#api_run = subprocess.Popen(get_api_path, creationflags=subprocess.CREATE_NEW_CONSOLE)
#time.sleep(5)

baseUrl = "http://localhost:5000"

Number_of_calculations = 100

# Path into folder with your python script and connection module
script_dir = Path(__file__).resolve().parent
project_file_path = script_dir / "LoadNearUnstiffenedEnd.ideaCon"

# Import cross-sections
css_list = read_xlsx('Cross-sections.xlsx')
steel_grade_list = ['S 235', 'S 275', 'S 355', 'S 450', 'S 420 MH/MLH', 'S 460 MH/MLH', 'S 620 Q/QL/QL1', 'S 690 Q/QL/QL1', 'S 500 MC', 'S 550 MC', 'S 600 MC', 'S 650 MC', 'S 700 MC']
parameter_list = np.unique(np.round(np.logspace(np.log10(0.05), np.log10(1), num=100), 3))
#parameter_list_naive_pyyhon = list(parameter_list)   # so random.choice works
#random.seed(42) #I would use this for reproducibility, if I want same results each time

start = time.time() 
# Create client attached to already running service
with connection_api_service_attacher.ConnectionApiServiceAttacher(baseUrl).create_api_client() as api_client:

        # Open project
        print("Opening project %s" % project_file_path)
            
        openedProject = api_client.project.open_project_from_filepath(project_file_path)
        activeProjectId = api_client.project.active_project_id

        # Get list of all connections in the project
        connections_in_project = api_client.connection.get_connections(api_client.project.active_project_id)

        # Iterate project items
        for connection in connections_in_project:
            print("Processing connection : %s" % connection)
            
            # Print header
            print('i; steel_grade; fy; fu; epsilon_lim; css; x; a; plate_strain; analysis; F_plRd_CBFEM; buckling; Fcr; alpha_cr') 

            # Run stress-strain CBFEM analysis for the first connection
            calc_params = ideastatica_connection_api.ConCalculationParameter()
            calc_params.connection_ids = [connection.id]
            conn_ID = calc_params.connection_ids[0]
            calc_params.analysis_type = 'buckling'
           
            for i in range(0, Number_of_calculations):
                index = i

                css = random.choice(css_list)
                steel_grade = random.choice(steel_grade_list)
                parameter = random.choice(list(parameter_list))

                cross_section = {
                    "new_cs_steel": steel_grade,
                    "new_cs": css,
                    "beam_name": 'M3'
                }
                change_cross_section(api_client.member, openedProject.project_id, connection.id, api_client.material,cross_section)

                x = parameter
                a = round(random.uniform(1, 5), 2)
                #cepp = 1


                idea_parameter_update = [
                    {
                        "key": "x",
                        "expression": str(parameter)
                    },
                    {
                        "key": "a",
                        "expression": str(a)
                    }, 
                ]
                # Get parameters
                api_response = api_client.parameter.get_parameters(activeProjectId,conn_ID)
                fy = api_response[1].value
                fu = api_response[2].value
                # Update parameters for the connection connectionId in the project projectId by values passed in parameters
                api_response = api_client.parameter.update(activeProjectId, conn_ID, idea_parameter_update=idea_parameter_update)

                #Calculate plastic strain limit
                epsilon_u = max(0.06, 0.6 * (1- fy / fu))
                epsilon_lim = 0.25 * epsilon_u
                # Get project settings
                setup = api_client.project.get_setup(activeProjectId)
                # Modify constant values in Project settings
                setup.limit_plastic_strain = epsilon_lim
                setup.hss_limit_plastic_strain = epsilon_lim
                setup.stop_at_limit_strain = True
                # Update Project Settings
                api_client.project.update_setup(activeProjectId, setup)
                setup = api_client.project.get_setup(activeProjectId)

                # Calculate project item
                results = api_client.calculation.get_raw_json_results(api_client.project.active_project_id, calc_params)
                results = json.loads(results[0])
                Load = api_client.load_effect.get_load_effects(activeProjectId,connection.id)
                AppliedLoad = Load[0].member_loadings[0].section_load.n
                analysis = results['summary']['1']['checkValue']

                # Update load effects if needed
                if analysis < 0.3:
                    load_effect:ConLoadEffect = Load[0]
                    load_effect.member_loadings[0].section_load.n = AppliedLoad*analysis*1.05
                    load_effect.member_loadings[1].section_load.n = AppliedLoad*analysis*1.05
                    api_client.load_effect.update_load_effect(activeProjectId,connection.id, Load[0])
                    results = api_client.calculation.get_raw_json_results(api_client.project.active_project_id, calc_params)
                    results = json.loads(results[0])
                    Load = api_client.load_effect.get_load_effects(activeProjectId,connection.id)
                    AppliedLoad = Load[0].member_loadings[0].section_load.n
                    analysis = results['summary']['1']['checkValue']

                                    
                while analysis == 1:
                    load_effect:ConLoadEffect = Load[0]
                    load_effect.member_loadings[0].section_load.n = AppliedLoad*2
                    load_effect.member_loadings[1].section_load.n = AppliedLoad*2
                    api_client.load_effect.update_load_effect(activeProjectId,connection.id, Load[0])
                    results = api_client.calculation.get_raw_json_results(api_client.project.active_project_id, calc_params)
                    results = json.loads(results[0])
                    Load = api_client.load_effect.get_load_effects(activeProjectId,connection.id)
                    AppliedLoad = Load[0].member_loadings[0].section_load.n
                    analysis = results['summary']['1']['checkValue']
                    

                plate_strain = results['summary']['2']['checkValue']
                buckling = results['bucklings']['0']['buckligLoadCoefficient']
                F_plRd_CBFEM = AppliedLoad*analysis/-1000
                Fcr = AppliedLoad * buckling /-1000 
                alpha_cr = Fcr / F_plRd_CBFEM
                
                # Print all results into one row   
                print(index, steel_grade, fy/1e6, fu/1e6, epsilon_lim, css, x, a, plate_strain, analysis, F_plRd_CBFEM, buckling, Fcr, alpha_cr, sep=';')

    

end = time.time()

print(f"Execution time: {end - start:.4f} seconds")


