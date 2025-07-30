import os
import argparse
import shutil
import time
from ultralytics import YOLO

def train_model(dataset_path):
    # Locate data.yaml
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    print("Using dataset from: {}".format(yaml_path))

    if not os.path.exists(yaml_path):
        print("Error: data.yaml not found at {}".format(yaml_path))
        return

    # Create "output" folder in current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO model
    model = YOLO('yolo11l.pt')  # Change this if needed

    # Train the model
    results = model.train(
        seed=4,
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=32,
        project=output_dir,
        name="yolo_11_windows",
        exist_ok=True,
        augment=True,
        shear=10,
        flipud=0.5,
        fliplr=0.7
    )

    # Get run path
    run_path = results.save_dir
    weights_dir = os.path.join(run_path, 'weights')
    best_weights = os.path.join(weights_dir, 'best.pt')
    final_weights = os.path.join(run_path, 'final.pt')

    # Copy best.pt to final.pt
    if os.path.exists(best_weights):
        shutil.copy(best_weights, final_weights)
        print(">>> Final model saved to: {}".format(final_weights))
    else:
        print(">>> best.pt not found. Check training status.")
        return

    # Evaluate the best model
    print("? Evaluating the model on the validation set...")
    best_model = YOLO(final_weights)
    metrics = best_model.val(data=yaml_path)

    # Print evaluation results
    print("\n? Evaluation results:")
    print("  mAP50:     {:.4f}".format(metrics.box.map50))
    print("  mAP50-95:  {:.4f}".format(metrics.box.map))
    print("  Precision: {:.4f}".format(metrics.box.precision))
    print("  Recall:    {:.4f}".format(metrics.box.recall))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model and evaluate it.")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset folder containing data.yaml')
    args = parser.parse_args()

    # Record start and end time
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    train_model(args.dataset_path)

    print("-- Training started at {}".format(start_time))
    end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print("-- Training completed at {}".format(end_time))
    print("-- Training process finished successfully!")
    print("-- Output directory: {}".format(os.path.join(os.getcwd(), 'output')))
    print("-- Check the output directory for results and logs.")
