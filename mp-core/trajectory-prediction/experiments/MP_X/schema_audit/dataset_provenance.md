### SECTION 1 — Dataset provenance

- Absolute path: `C:\Users\OWNER\Desktop\new_datasets\Barcelona_v3_manual_master_dataset\model_C_dataset.csv`
- Rows: **1,064,379**  ·  Tracks: **3,534**  ·  world coords: **master**
- Recordings: ['esplanade_espanya_01', 'placa_catalunya_01', 'placa_espanya_01', 'red_bridge_combined_01', 'stairs_montjuic_01']
- Splits: ['test', 'train', 'val']
- Columns: ['recording_id', 'trajectory_id', 'timestep', 'split', 'u', 'v', 'du', 'dv', 'speed', 'heading_sin', 'heading_cos', 'turn_rate', 'dist_to_obstacle_norm', 'dist_to_boundary_norm', 'target_du', 'target_dv', 'src_track_id', 'frame', 'world_x', 'world_y', 'dist_to_obstacle_v3_m', 'dist_to_walkable_boundary_v3_m', 'inbounds_v3', 'world_x_p1', 'world_x_n1', 'world_x_n2', 'world_y_p1', 'world_y_n1', 'world_y_n2', 'du_back', 'dv_back', 'du_fwd', 'dv_fwd', 'du_n2', 'dv_n2']
- Missing values: {'world_x_p1': 3534, 'world_x_n1': 3534, 'world_x_n2': 7068, 'world_y_p1': 3534, 'world_y_n1': 3534, 'world_y_n2': 7068, 'du_back': 3534, 'dv_back': 3534, 'du_fwd': 3534, 'dv_fwd': 3534, 'du_n2': 7068, 'dv_n2': 7068}
- Duplicate full rows: 0  ·  Duplicate keys (rec+track+timestep): 0
- Non-contiguous timestep tracks: 0 (timestep is a per-track 0..n-1 index; raw `frame` may legitimately have gaps)
- trajectory_id spanning >1 recording: 0 (IDs are recording-prefixed)
- Status: **PASS** 

Outputs: `provenance_table.csv`.
