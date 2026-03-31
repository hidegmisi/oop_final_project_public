from pathlib import Path

from data_import.definitions import DatasetDefinition, ExtraImageSetDefinition


def build_dataset_configs(project_root: Path) -> list[DatasetDefinition]:
    """Return the list of dataset download configurations."""
    return [
        DatasetDefinition(
            slug="iamtapendu/chest-x-ray-lungs-segmentation",
            dest=project_root / "data" / "raw" / "chest_xray_lungs",
            metadata_filename="MetaData.csv",
            metadata_keywords=["metadata"],
            metadata_key_candidates=["id", "image", "image_id", "imageid", "image_name",
                                     "filename", "file"],
            image_markers=["image/"],
            mask_markers=["mask/"],
            subset_metadata="MetaData_subset.csv",
        ),
        DatasetDefinition(
            slug="farjanakabirsamanta/skin-cancer-dataset",
            dest=project_root / "data" / "raw" / "skin_cancer",
            metadata_filename="HAM10000_metadata.csv",
            metadata_keywords=["ham10000", "metadata"],
            metadata_key_candidates=["image_id", "image", "imageid", "filename"],
            image_markers=[],
            mask_markers=None,
            subset_metadata="HAM10000_metadata_subset.csv",
        ),
        DatasetDefinition(
            slug="iamtapendu/rsna-pneumonia-processed-dataset",
            dest=project_root / "data" / "raw" / "rsna_pneumonia",
            use_images_subset=True,
            key_column="patientId",
            image_markers=["training/images/"],
            mask_markers=["training/masks/"],
            subset_metadata="stage2_train_metadata_subset.csv",
            extra_sets=[
                ExtraImageSetDefinition(
                    image_markers=["test/"],
                    mask_markers=None,
                    metadata_filename="stage2_test_metadata.csv",
                    metadata_keywords=["stage2", "test", "metadata"],
                    metadata_key_candidates=["patientid", "patient_id", "patientId",
                                             "image", "image_id", "filename"],
                    subset_metadata="stage2_test_metadata_subset.csv",
                )
            ],
        ),
    ]
