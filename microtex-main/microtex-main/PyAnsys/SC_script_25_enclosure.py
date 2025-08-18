# Python Script, API Version = V232

# Open File
importOptions = ImportOptions.Create()
DocumentOpen.Execute(r"E:\Summer Placement\Summer Placement Testbed\model8.step", importOptions)
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


# Create Enclosure
selection = BodySelection.Create(GetRootPart().Bodies[0])

options = EnclosureOptions()
options.EnclosureType = EnclosureType.Box
options.EnclosureCushion = BoxEnclosureCushion(MM(0),MM(0),MM(5),MM(5),MM(0),MM(5))
options.CushionProportion = PERCENT(0)
result = Enclosure.Create(selection, options)
# EndBlock


# Extrude 2 Faces
selection = Selection.CreateByObjects([RayFire.Fire(Point.Create(MM(303.617782471932), MM(11.1873544573764), MM(-444.358067800937)),
        Direction.Create(-0.562800002076919, -0.048624861480512, 0.825161669315911),
        4.06760320478719E-05, 0.000203380160239576)[1],
    RayFire.Fire(Point.Create(MM(303.009547318857), MM(40.8913969600853), MM(-443.022523206863)),
        Direction.Create(-0.562800002076919, -0.048624861480512, 0.825161669315911),
        4.06760320478719E-05, 0.000203380160239526)[1]])
options = ExtrudeFaceOptions()
options.ExtrudeType = ExtrudeType.Cut
result = ExtrudeFaces.Execute(selection, MM(-1), options)
# EndBlock


# Exclude items from physics
selection = BodySelection.Create(GetRootPart().Bodies[0])
suppress = True
ViewHelper.SetSuppressForPhysics(selection, suppress)
# EndBlock

# Change Object Visibility
selection = BodySelection.Create(GetRootPart().Bodies[0])
visibility = VisibilityType.Hide
inSelectedView = False
faceLevel = False
ViewHelper.SetObjectVisibility(selection, visibility, inSelectedView, faceLevel)
# EndBlock


# Create Named Selection Group
primarySelection = Selection.Create(RayFire.Fire(Point.Create(MM(381.980942850467), MM(-355.881133692647), MM(144.848722287511)),
    Direction.Create(-0.720435429439982, 0.639715119387482, -0.267838305764296),
    4.06760320478677E-05, 0.000203380160239536)[1])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "inlet")
# EndBlock


# Create Named Selection Group
primarySelection = Selection.Create(RayFire.Fire(Point.Create(MM(249.524962381345), MM(478.116476395787), MM(39.6919378492687)),
    Direction.Create(-0.473904954947072, -0.877894570477991, -0.0686674362553154),
    3.0803867472452E-05, 0.000154019337362354)[1])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "outlet")
# EndBlock


# Create Named Selection Group
primarySelection = Selection.Create(RayFire.Fire(Point.Create(MM(166.545726433212), MM(-452.486524733071), MM(247.383954588008)),
    Direction.Create(-0.302924216167066, 0.844173457478959, -0.442276037048798),
    3.08038674724793E-05, 0.000154019337362383)[1])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "moving_wall")
# EndBlock

# Create Named Selection Group
primarySelection = Selection.CreateByObjects([RayFire.Fire(Point.Create(MM(327.930927900763), MM(-339.820051560354), MM(266.740443317613)),
        Direction.Create(-0.597228477626512, 0.641187900970292, -0.481867431106554),
        3.08038674725175E-05, 0.000154019337362375)[1],
    RayFire.Fire(Point.Create(MM(-212.712509384443), MM(-484.240989699978), MM(110.442820802378)),
        Direction.Create(0.393283574726255, 0.898384061319412, -0.195535439799996),
        3.08038674724641E-05, 0.000154019337362371)[1]])
secondarySelection = Selection.Empty()
result = NamedSelection.Create(primarySelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "symmetry")
# EndBlock

# This bit of code selects every face, then deselects every face we don't want
# Create Named Selection Group
primarySelection = FaceSelection.Create(GetRootPart().Components[0].Content.Bodies[0].Faces[:])
secondarySelection = Selection.Empty()
# Select the faces to subtract
subtractSelection = Selection.CreateByObjects([RayFire.Fire(Point.Create(MM(402.055769407651), MM(-232.930525276202), MM(-273.571302304827)),
        Direction.Create(-0.753055459450827, 0.404209618315208, 0.519155140061983),
        3.08038674724817E-05, 0.000154019337362403)[1],
    RayFire.Fire(Point.Create(MM(406.276286487217), MM(-221.338126280184), MM(-276.475011868186)),
        Direction.Create(-0.753055459450827, 0.404209618315208, 0.519155140061983),
        3.08038674724817E-05, 0.000154019337362386)[1],
    RayFire.Fire(Point.Create(MM(377.691462240609), MM(374.992281832674), MM(-92.6772255744093)),
        Direction.Create(-0.713205955643705, -0.677156805323579, 0.181096454516136),
        3.08038674725449E-05, 0.000154019337362481)[1],
    RayFire.Fire(Point.Create(MM(-116.737921834505), MM(527.001088223437), MM(0.582293605502641)),
        Direction.Create(0.218195402129986, -0.975897826223945, 0.00376819064166791),
        3.08038674724739E-05, 0.000154019337362381)[1],
    RayFire.Fire(Point.Create(MM(143.508335112248), MM(512.408447477331), MM(96.9394572496032)),
        Direction.Create(-0.26782659108565, -0.948273695681988, -0.170428621965514),
        3.08038674725048E-05, 0.000154019337362356)[1]])
diffSelection = FaceSelection.Difference(primarySelection, subtractSelection)
result = NamedSelection.Create(diffSelection, secondarySelection)
# EndBlock

# Rename Named Selection
result = NamedSelection.Rename("Group1", "wall")
# EndBlock

# Save File
options = ExportOptions.Create()
DocumentSave.Execute(r"E:\Summer Placement\Summer Placement Testbed\automationtest_25_enclosure.scdoc", options)
# EndBlock