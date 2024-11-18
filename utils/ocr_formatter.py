import numpy as np
from typing import List, Dict, Any, Tuple


def calculate_page_dimensions(
    ocr_data: List[Dict[str, Any]], char_aspect_ratio: float = 0.5
) -> Tuple[int, int]:
    """
    Calculate the required page dimensions based on the OCR data.

    Args:
        ocr_data: List of OCR entries containing words and their geometries
        char_aspect_ratio: Width to height ratio of a single character (default 0.5)
                         Used to maintain reasonable text proportions

    Returns:
        Tuple of (width, height) in characters
    """
    max_x, max_y = 0, 0
    min_x, min_y = float("inf"), float("inf")

    for line in ocr_data:
        geometry = line["geometry"]
        if isinstance(geometry, np.ndarray):
            geometry = geometry.tolist()

        # Extract x and y coordinates
        x_coords = [coord[0] for coord in geometry]
        y_coords = [coord[1] for coord in geometry]

        max_x = max(max_x, max(x_coords))
        max_y = max(max_y, max(y_coords))
        min_x = min(min_x, min(x_coords))
        min_y = min(min_y, min(y_coords))

    # Calculate relative document dimensions
    doc_width = max_x - min_x
    doc_height = max_y - min_y

    # Convert to character-based dimensions
    # Scale the width based on the aspect ratio to maintain reasonable proportions
    char_width = int(
        100 * doc_width / char_aspect_ratio
    )  # Base width of 100 characters
    char_height = int(100 * doc_height)

    return char_width, char_height


def normalize_coordinates(ocr_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize coordinates to start from (0,0) and adjust all coordinates accordingly.

    Args:
        ocr_data: List of OCR entries containing words and their geometries

    Returns:
        OCR data with normalized coordinates
    """
    # Find minimum x and y coordinates
    min_x = float("inf")
    min_y = float("inf")

    for line in ocr_data:
        geometry = line["geometry"]
        if isinstance(geometry, np.ndarray):
            geometry = geometry.tolist()

        x_coords = [coord[0] for coord in geometry]
        y_coords = [coord[1] for coord in geometry]

        min_x = min(min_x, min(x_coords))
        min_y = min(min_y, min(y_coords))

    # Create deep copy and normalize coordinates
    normalized_data = []
    for line in ocr_data:
        new_line = dict(line)
        geometry = line["geometry"]
        if isinstance(geometry, np.ndarray):
            geometry = geometry.tolist()

        new_geometry = [[coord[0] - min_x, coord[1] - min_y] for coord in geometry]

        if isinstance(line["geometry"], np.ndarray):
            new_line["geometry"] = np.array(new_geometry, dtype=np.float32)
        else:
            new_line["geometry"] = new_geometry

        normalized_data.append(new_line)

    return normalized_data


def create_text_layout(
    ocr_data: List[Dict[str, Any]],
    page_width: int = None,
    page_height: int = None,
    char_aspect_ratio: float = 0.5,
) -> str:
    """
    Convert OCR data into a text-based representation preserving spatial layout.

    Args:
        ocr_data: List of OCR entries containing words and their geometries
        page_width: Optional override for calculated page width
        page_height: Optional override for calculated page height
        char_aspect_ratio: Width to height ratio of a single character

    Returns:
        String representation of the document preserving spatial layout
    """
    # Normalize coordinates to start from (0,0)
    normalized_data = normalize_coordinates(ocr_data)

    # Calculate dimensions if not provided
    if page_width is None or page_height is None:
        calculated_width, calculated_height = calculate_page_dimensions(
            normalized_data, char_aspect_ratio
        )
        page_width = page_width or calculated_width
        page_height = page_height or calculated_height

    # Initialize empty text grid
    text_grid = [[" " for _ in range(page_width)] for _ in range(page_height)]

    # Process each line
    for line in normalized_data:
        geometry = line["geometry"]
        if isinstance(geometry, np.ndarray):
            geometry = geometry.tolist()

        # Calculate positions
        x_coords = [coord[0] for coord in geometry]
        y_coords = [coord[1] for coord in geometry]

        # Convert normalized coordinates to grid positions
        grid_x = int(min(x_coords) * (page_width - 1))
        grid_y = int(min(y_coords) * (page_height - 1))

        # Get text content
        if "words" in line:
            text = " ".join(word["value"] for word in line["words"])
        else:
            text = line.get("value", "")

        # Place text in grid
        for i, char in enumerate(text):
            if grid_x + i < page_width and grid_y < page_height:
                text_grid[grid_y][grid_x + i] = char

    return "\n".join("".join(row) for row in text_grid)


def format_document(
    ocr_data: List[Dict[str, Any]], compact: bool = True, char_aspect_ratio: float = 0.5
) -> str:
    """
    Format the document with option to remove empty lines and trailing spaces.

    Args:
        ocr_data: List of OCR entries containing words and their geometries
        compact: If True, removes excessive empty lines and trailing spaces
        char_aspect_ratio: Width to height ratio of a single character

    Returns:
        Formatted string representation of the document
    """
    # Create the full layout
    text = create_text_layout(ocr_data, char_aspect_ratio=char_aspect_ratio)

    if compact:
        # Split into lines
        lines = text.split("\n")
        # Remove trailing spaces from each line
        lines = [line.rstrip() for line in lines]
        # Remove empty lines at the beginning and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        # Collapse multiple empty lines into one
        result = []
        prev_empty = False
        for line in lines:
            if line.strip():
                result.append(line)
                prev_empty = False
            elif not prev_empty:
                result.append("")
                prev_empty = True
        return "\n".join(result)

    return text
