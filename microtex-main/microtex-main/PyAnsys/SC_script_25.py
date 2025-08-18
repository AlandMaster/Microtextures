#This script uses Ray selection to detect faces, using vectors (explicit mouse locations).
#It also uses the older method of creating and merging shapes, and boolean subtraction using the combine function.

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
selection = Selection.Create(RayFire.Fire(Point.Create(MM(413.208129176094), MM(-234.757136814525), MM(257.060933253314)),
    Direction.Create(-0.758866518358856, 0.430539594594472, -0.488627941076168),
    2.13915746337072E-05, 0.000106957873168372)[1])
options = ExtrudeFaceOptions()
options.ExtrudeType = ExtrudeType.ForceAdd
result = ExtrudeFaces.Execute(selection, MM(-1.01), options)
# EndBlock


# Set Sketch Plane
sectionPlane = Plane.Create(Frame.Create(Point.Create(MM(-0.000698949144037528), MM(0.500698952479627), MM(0.510000003335324)), 
    Direction.DirX, 
    -Direction.DirZ))
result = ViewHelper.SetSketchPlane(sectionPlane, None)
# EndBlock

# Sketch Rectangle
point1 = Point2D.Create(MM(-2.00000000000003),MM(-18))
point2 = Point2D.Create(MM(2.00000000000004),MM(-18))
point3 = Point2D.Create(MM(2.00000000000004),MM(18))
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
result = ExtrudeFaces.Execute(selection, MM(5), options)
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
result = Move.Translate(selection, direction, MM(13), options)
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
primarySelection = Selection.Create(RayFire.Fire(Point.Create(MM(229.221236984557), MM(266.490756163672), MM(425.717676935456)),
    Direction.Create(-0.433012701892219, -0.5, -0.75),
    2.13915746336393E-05, 0.000106957873168305)[1])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "inlet")
# EndBlock


# Create Named Selection Group
primarySelection = Selection.Create(RayFire.Fire(Point.Create(MM(107.003355035983), MM(115.344939758134), MM(-503.831121412681)),
    Direction.Create(-0.202690280930401, -0.215020947943432, 0.95534425311605),
    2.1391574633685E-05, 0.000106957873168332)[1])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "outlet")
# EndBlock


# Create Named Selection Group
primarySelection = Selection.Create(RayFire.Fire(Point.Create(MM(316.493840943628), MM(173.799562738804), MM(416.880496814734)),
    Direction.Create(-0.585876376083365, -0.31324336472073, -0.747413852163485),
    2.13915746336105E-05, 0.000106957873168327)[1])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "moving_wall")
# EndBlock


# Create Named Selection Group
primarySelection = Selection.CreateByObjects([RayFire.Fire(Point.Create(MM(224.657881692857), MM(171.030989223283), MM(475.200735031706)),
        Direction.Create(-0.412263142209646, -0.311596379204029, -0.856123120843239),
        2.13915746336185E-05, 0.000106957873168276)[1],
    RayFire.Fire(Point.Create(MM(-136.369360355949), MM(170.357050762787), MM(508.020682973097)),
        Direction.Create(0.252183856370473, -0.30804208486362, -0.917340382049646),
        2.13915746336741E-05, 0.000106957873168331)[1]])
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
subtractSelection = Selection.CreateByObjects([RayFire.Fire(Point.Create(MM(224.657881692857), MM(171.030989223283), MM(475.200735031706)),
        Direction.Create(-0.412263142209646, -0.311596379204029, -0.856123120843239),
        2.13915746336185E-05, 0.000106957873168276)[1],
    RayFire.Fire(Point.Create(MM(-136.369360355949), MM(170.357050762787), MM(508.020682973097)),
        Direction.Create(0.252183856370473, -0.30804208486362, -0.917340382049646),
        2.13915746336741E-05, 0.000106957873168331)[1],
    RayFire.Fire(Point.Create(MM(316.493840943628), MM(173.799562738804), MM(416.880496814734)),
        Direction.Create(-0.585876376083365, -0.31324336472073, -0.747413852163485),
        2.13915746336105E-05, 0.000106957873168327)[1],
    RayFire.Fire(Point.Create(MM(107.003355035983), MM(115.344939758134), MM(-503.831121412681)),
        Direction.Create(-0.202690280930401, -0.215020947943432, 0.95534425311605),
        2.1391574633685E-05, 0.000106957873168332)[1],
    RayFire.Fire(Point.Create(MM(229.221236984557), MM(266.490756163672), MM(425.717676935456)),
        Direction.Create(-0.433012701892219, -0.5, -0.75),
        2.13915746336393E-05, 0.000106957873168305)[1]]
        )
diffSelection = FaceSelection.Difference(primarySelection, subtractSelection)
result = NamedSelection.Create(diffSelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "wall")
# EndBlock

# Save File
options = ExportOptions.Create()
DocumentSave.Execute(r"E:\Summer Placement\Summer Placement Testbed\automationtest_25.scdoc", options)
# EndBlock