import os
import zipfile
import glob
import xarray as xr

# Create a folder to hold the final, unified .nc files
extract_dir = "ERA5_UAE_Unified_Data"
temp_dir = "temp_extraction"
os.makedirs(extract_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

# Find all the zip files (even if they have a .nc extension)
downloaded_files = glob.glob("ERA5_UAE_*.nc")

print(f"Found {len(downloaded_files)} files. Starting extraction and merge process...")

for file_path in downloaded_files:
    if zipfile.is_zipfile(file_path):
        filename_base = os.path.basename(file_path)
        print(f"\nProcessing archive: {filename_base}")
        
        # 1. Extract the split files into the temporary folder
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # 2. Locate the extracted accum and instant files
        accum_file = os.path.join(temp_dir, "data_stream-oper_stepType-accum.nc")
        instant_file = os.path.join(temp_dir, "data_stream-oper_stepType-instant.nc")
        
        # Check if both files exist before attempting to merge
        if os.path.exists(accum_file) and os.path.exists(instant_file):
            # 3. Merge the datasets
            ds_accum = xr.open_dataset(accum_file)
            ds_instant = xr.open_dataset(instant_file)
            ds_merged = xr.merge([ds_accum, ds_instant])
            
            # 4. Save the unified file to the final directory
            final_output_path = os.path.join(extract_dir, filename_base)
            ds_merged.to_netcdf(final_output_path)
            print(f"--> Merged successfully into {final_output_path}")
            
            # Close datasets to free memory
            ds_accum.close()
            ds_instant.close()
            ds_merged.close()
            
            # 5. Clean up the temporary files so they are ready for the next zip loop
            os.remove(accum_file)
            os.remove(instant_file)
            
        else:
            print(f"--> Warning: Expected accum/instant files not found in {filename_base}")
            
    else:
        # If it's already a standard, unzipped .nc file, just move it
        print(f"\n{file_path} is already a unified NetCDF. Moving to folder...")
        os.rename(file_path, os.path.join(extract_dir, os.path.basename(file_path)))

# Clean up the empty temporary directory
os.rmdir(temp_dir)
print(f"\nAll files successfully merged and extracted to the '{extract_dir}' folder!")