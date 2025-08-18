# Python Script, API Version = V232
fileName = r"E:\Summer Placement\Summer Placement Testbed\model8.step"

# Open File
importOptions = ImportOptions.Create()
DocumentOpen.Execute(fileName, importOptions)
# EndBlock

# Rotate About X Handle
selection = BodySelection.Create(GetRootPart().Bodies[0])
anchorPoint = Move.GetAnchorPoint(selection)
axis = Line.Create(anchorPoint, Direction.DirX)
options = MoveOptions()
result = Move.Rotate(selection, axis, DEG(-90), options)
# EndBlock

# Copy items
result = Copy.Execute(BodySelection.Create(GetRootPart().Bodies[0]))
# EndBlock

# Change Object Visibility
selection = BodySelection.Create(GetRootPart().Bodies[0])
visibility = VisibilityType.Hide
inSelectedView = False
faceLevel = False
ViewHelper.SetObjectVisibility(selection, visibility, inSelectedView, faceLevel)
# EndBlock


# Extrude 1 Face
selection = FaceSelection.Create(GetRootPart().Bodies[1].Faces[1])
reference = Selection.Create(GetRootPart().Bodies[1].Edges[95].GetChildren[ICurvePoint]()[1])
options = ExtrudeFaceOptions()
options.ExtrudeType = ExtrudeType.ForceAdd
result = ExtrudeFaces.SetDimension(selection, reference, MM(4.28612223837642E-17), options)
# EndBlock


# Set Sketch Plane
sectionPlane = Plane.Create(Frame.Create(Point.Create(MM(-0.000698949144037528), MM(0.500698952479627), MM(0.510000003335324)), 
    Direction.DirX, 
    -Direction.DirZ))
result = ViewHelper.SetSketchPlane(sectionPlane, None)
# EndBlock

# Sketch Rectangle
point1 = Point2D.Create(MM(-2.00000000000003),MM(-6))
point2 = Point2D.Create(MM(2.00000000000004),MM(-6))
point3 = Point2D.Create(MM(2.00000000000004),MM(6))
result = SketchRectangle.Create(point1, point2, point3)
# EndBlock

# Change Object Visibility
selection = BodySelection.Create(GetRootPart().Bodies[1])
visibility = VisibilityType.Hide
inSelectedView = False
faceLevel = False
ViewHelper.SetObjectVisibility(selection, visibility, inSelectedView, faceLevel)
# EndBlock

# Solidify Sketch
mode = InteractionMode.Solid
result = ViewHelper.SetViewMode(mode, None)
# EndBlock

# Extrude 1 Face
selection = FaceSelection.Create(GetRootPart().Bodies[2].Faces[0])
options = ExtrudeFaceOptions()
options.ExtrudeType = ExtrudeType.Add
result = ExtrudeFaces.Execute(selection, MM(1.5), options)
# EndBlock

# Change Object Visibility
selection = BodySelection.Create(GetRootPart().Bodies[1])
visibility = VisibilityType.Show
inSelectedView = False
faceLevel = False
ViewHelper.SetObjectVisibility(selection, visibility, inSelectedView, faceLevel)
# EndBlock

# Merge Bodies
targets = BodySelection.Create([GetRootPart().Bodies[2],
    GetRootPart().Bodies[1]])
result = Combine.Merge(targets)
# EndBlock

# Change Object Visibility
selection = BodySelection.Create(GetRootPart().Bodies[0])
visibility = VisibilityType.Show
inSelectedView = False
faceLevel = False
ViewHelper.SetObjectVisibility(selection, visibility, inSelectedView, faceLevel)
# EndBlock

# Intersect Bodies
targets = BodySelection.Create(GetRootPart().Bodies[1])
tools = BodySelection.Create(GetRootPart().Bodies[0])
options = MakeSolidsOptions()
result = Combine.Intersect(targets, tools, options)
# EndBlock


# Delete Objects
selection = BodySelection.Create([GetRootPart().Bodies[0],
    GetRootPart().Bodies[2]])
result = Delete.Execute(selection)
# EndBlockSelection.Empty()

# Translate Along Y Handle
selection = BodySelection.Create(GetRootPart().Bodies[0])
direction = Direction.DirY
options = MoveOptions()
result = Move.Translate(selection, direction, MM(-0.5), options)
# EndBlock

# Translate Along Z Handle
selection = BodySelection.Create(GetRootPart().Bodies[0])
direction = Direction.DirZ
options = MoveOptions()
result = Move.Translate(selection, direction, MM(5.5), options)
# EndBlock

# Create Datum Plane
selection = Selection.Create(GetRootPart().CoordinateSystems[0].Axes[0])
result = DatumPlaneCreator.Create(selection, False, None)
# EndBlock

# Slice Bodies by Plane
selection = BodySelection.Create(GetRootPart().Bodies[0])
datum = Selection.Create(GetRootPart().DatumPlanes[0])
result = SplitBody.ByCutter(selection, datum)
# EndBlock

# Delete Objects
selection = BodySelection.Create(GetRootPart().Bodies[0])
result = Combine.RemoveRegions(selection)
# EndBlock

# Create Named Selection Group
primarySelection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[5])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "inlet")
# EndBlock

# Create Named Selection Group
primarySelection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[4])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "outlet")
# EndBlock

# Create Named Selection Group
primarySelection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[3])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "moving wall")
# EndBlock

# Create Named Selection Group
primarySelection = FaceSelection.Create([GetRootPart().Bodies[0].Faces[1],
    GetRootPart().Bodies[0].Faces[2]])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "symmetry")
# EndBlock

# Create Named Selection Group
primarySelection = FaceSelection.Create(GetRootPart().Bodies[0].Faces[:])
secondarySelection = Selection.Empty()
# Select the faces to subtract
subtractSelection = FaceSelection.Create([GetRootPart().Bodies[0].Faces[1],
    GetRootPart().Bodies[0].Faces[2],
    GetRootPart().Bodies[0].Faces[3],
    GetRootPart().Bodies[0].Faces[4],
    GetRootPart().Bodies[0].Faces[5]])
diffSelection = FaceSelection.Difference(primarySelection, subtractSelection)
result = NamedSelection.Create(diffSelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "wall")
# EndBlock

# Save File
options = ExportOptions.Create()
DocumentSave.Execute(r"E:\Summer Placement\Summer Placement Testbed\automationtest.scdoc", options)
# EndBlock