import vtk
import numpy as np

x_thresh = [-1.1, 2.1]*10
y_thresh = [1.5, 4.5]*10
z_thresh = [-0.4, 2.1]*10


class VtkPointCloud:
    def __init__(self, zMin=-1.0, zMax=1.0, maxNumPoints=1e6):
        self.init_planes()
        self.init_points(zMin, zMax, maxNumPoints)

    def addPoint(self, point, color_num):
        if self.vtkPoints.GetNumberOfPoints() < self.maxNumPoints:
            pointId = self.vtkPoints.InsertNextPoint(point[:])
            self.vtkDepth.InsertNextValue(color_num)
            self.vtkCells.InsertNextCell(1)
            self.vtkCells.InsertCellPoint(pointId)
        else:
            r = np.random.randint(0, self.maxNumPoints)
            self.vtkPoints.SetPoint(r, point[:])
        self.vtkCells.Modified()
        self.vtkPoints.Modified()
        self.vtkDepth.Modified()

    def addPlane(self, plane_center, normal, x_axis, y_axis):
        self.vtkPlanes.SetCenter(plane_center)
        self.vtkPlanes.SetNormal(normal)
        self.vtkPlanes.SetPoint1(x_axis)
        self.vtkPlanes.SetPoint2(y_axis)

    def init_points(self, zMin=-1.0, zMax=1.0, maxNumPoints=1e6):
        self.maxNumPoints = maxNumPoints
        self.vtkPolyData = vtk.vtkPolyData()
        self.vtkPoints = vtk.vtkPoints()
        self.vtkCells = vtk.vtkCellArray()
        self.vtkDepth = vtk.vtkDoubleArray()

        self.vtkDepth.SetName('DepthArray')
        self.vtkPolyData.SetPoints(self.vtkPoints)
        self.vtkPolyData.SetVerts(self.vtkCells)
        self.vtkPolyData.GetCellData().SetScalars(self.vtkDepth)
        self.vtkPolyData.GetCellData().SetActiveScalars('DepthArray')
        point_mapper = vtk.vtkPolyDataMapper()
        point_mapper.SetInputDataObject(self.vtkPolyData)
        point_mapper.SetColorModeToDefault()
        point_mapper.SetScalarRange(zMin, zMax)
        self.point_vtkActor = vtk.vtkActor()
        self.point_vtkActor.SetMapper(point_mapper)

    def init_planes(self):
        self.vtkPlanes = vtk.vtkPlaneSource()
        plane_mapper = vtk.vtkPolyDataMapper()
        plane_mapper.SetInputDataObject(self.vtkPlanes.GetOutput())
        self.plane_vtkActor = vtk.vtkActor()
        self.plane_vtkActor.SetMapper(plane_mapper)
        
def vtk_visualize(point_list):
    global x_thresh, y_thresh, z_thresh
    point_cloud = VtkPointCloud()

    for i in range(len(point_list)):
        point_coords = point_list[i]

        if (point_coords[0] > x_thresh[0]) and (point_coords[0] < x_thresh[1]) and \
                (point_coords[1] > y_thresh[0]) and (point_coords[1] < y_thresh[1]) and \
                (point_coords[2] > z_thresh[0]) and (point_coords[2] < z_thresh[1]):
            color_num = 0.7
        else:
            color_num = -1
        point_cloud.addPoint(point_list[i], color_num)

    # Add the velodyne plane
    for x in np.linspace(-4, 4, 100):
        for y in np.linspace(0, 2, 25):
            tmp_coords = np.array([x, 0, y])
            point_cloud.addPoint(tmp_coords, 1)
    # Add the floor plane
    plane_center = (-4, -4, -0.55)
    normal = (0, 0, 1)
    point1 = ([-4, 10, -0.55])
    point2 = ([4, -4, -0.55])
    point_cloud.addPlane(plane_center, normal, point1, point2)

    # Renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(point_cloud.point_vtkActor)
    renderer.AddActor(point_cloud.plane_vtkActor)

    renderer.SetBackground(0.0, 0.0, 0.0)
    renderer.ResetCamera()

    # Render Window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    # Interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
    render_window_interactor.SetRenderWindow(render_window)

    '''Add camera coordinates'''
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOutlineColor(0.9300, 0.5700, 0.1300)
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(render_window_interactor)
    widget.SetViewport(0.0, 0.0, 0.4, 0.4)
    widget.SetEnabled(1)
    widget.InteractiveOn()
    render_window.Render()
    render_window_interactor.Start()