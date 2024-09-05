import subprocess


def run_data_generation():
    # Define the arguments
    cube_length = 3
    back_max = 10
    data_dir = "Datasets"  # Adjust this path according to your project structure
    num_per_file = int(1e5)
    num_files = 10
    num_procs = 10
    start_idx = 0

    # Build the command
    command = [
        "python", "Scripts/dataset_generation.py",
        "--cube_length", str(cube_length),
        "--back_max", str(back_max),
        "--data_dir", data_dir,
        "--num_per_file", str(num_per_file),
        "--num_files", str(num_files),
        "--num_procs", str(num_procs),
        "--start_idx", str(start_idx)
    ]

    # Run the command
    subprocess.run(command)


if __name__ == "__main__":
    run_data_generation()
