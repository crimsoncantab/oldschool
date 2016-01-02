function varargout = pa2gui(varargin)
% PA2GUI MATLAB code for pa2gui.fig
%      PA2GUI, by itself, creates a new PA2GUI or raises the existing
%      singleton*.
%
%      H = PA2GUI returns the handle to a new PA2GUI or the handle to
%      the existing singleton*.
%
%      PA2GUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in PA2GUI.M with the given input arguments.
%
%      PA2GUI('Property','Value',...) creates a new PA2GUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before pa2gui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to pa2gui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help pa2gui

% Last Modified by GUIDE v2.5 27-Apr-2012 20:31:35

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @pa2gui_OpeningFcn, ...
                   'gui_OutputFcn',  @pa2gui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before pa2gui is made visible.
function pa2gui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to pa2gui (see VARARGIN)

% Choose default command line output for pa2gui
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes pa2gui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = pa2gui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;


% --- Executes on button press in pushbutton1.
function pushbutton1_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get image A from file upload
[FileName, PathName] = uigetfile({'*.png; *.jpg; *.bmp; *.gif; *.tif', 'All Image Files (*.png, *.jpg, *.bmp, *.gif, *.tif)'}, 'Choose an imamge file');
if isequal(FileName, 0)
    disp('User did not upload an image')
else
    disp(['User selected', fullfile(PathName, FileName)])
    A = imread(fullfile(PathName, FileName));
    handles.image = A;
    handles.imageurl = fullfile(PathName, FileName);
    axes(handles.axes1);
    imshow(handles.image, []);
    guidata(hObject, handles);
end


function A_output = CSH_level(level, A, B, mask, constmask, CSH_w, CSH_i, CSH_k, handles)

[hB wB dB] = size(B);

% w=32;
w = min(max(2^level,8),32);
if level > 0,
%     next_A = imresize(A, 1/2, 'nearest');
    next_A = imresize(A, 1/2);
    next_B = next_A;
    next_mask = imresize(mask, 1/2);
    next_constmask = imresize(constmask, 1/2);
    
    next_mask(next_mask ~= 0) = 1;
    next_constmask(next_constmask ~= 0) = 1;
    next_constmask(next_mask) = 1;
    disp('resize');
    
    A_temp = CSH_level(level - 1, next_A, next_B, next_mask, next_constmask, CSH_w, CSH_i, CSH_k, handles);
    
%     A = imresize(A_temp, [hB, wB], 'nearest');
    A = imresize(A_temp, [hB, wB]);
    A(repmat(mask, [1,1,dB]) == 0) = B(repmat(mask, [1,1,dB]) == 0);
    disp('upsample');
    handles.image = A;
    imshow(handles.image);
    drawnow;
%     pause;
    
    disp(level);
    A_output = CSH_fill(A, B, mask, constmask, w, CSH_i, CSH_k, handles);
    
else
    disp(level);
    A_output = CSH_fill(A, B, mask, constmask, w, CSH_i, CSH_k, handles);
end


function A_output = CSH_fill(A, B, mask, constmask, CSH_w, CSH_i, CSH_k, handles)
% CSH_w width
% CSH_i iterations
% CSH_k number of nearest neighbors

[hB wB dB] = size(B);

d = CSH_w - 1;

[nonrow, noncol] = find(mask);
xmin = max(min(noncol) - d, 1);
xmax = min(max(noncol), wB - d);
ymin = max(min(nonrow) - d, 1);
ymax = min(max(nonrow), hB - d);
disp('filling...');
for loops = 1:40
    disp(loops);
    NN = CSH_nn(A, B, CSH_w, CSH_i, CSH_k, 0, constmask);
%     NN = CSH_nn(A, B, CSH_w, CSH_i, CSH_k, 0, mask);
    votes = zeros(size(A));
    weights = zeros(size(A));
    A_orig = A;
    for i = ymin:ymax
        for j = xmin:xmax
           b_i = NN(i,j,2);
           b_j = NN(i,j,1);
           d_i = min(hB - b_i, d);
           d_j = min(wB - b_j, d);
           mask_w = sum(sum(mask(i:i+d_i,j:j+d_j)));
           if (mask_w > 0)
               B_p = B(b_i:b_i+d_i,b_j:b_j+d_j,1:dB);
               v_p = votes(i:i+d_i,j:j+d_j,1:dB);
               dif = double(A(i:i+d_i,j:j+d_j,1:dB)) - double(B_p);
%                weight = exp(double(d * d / (d_i * d_j * (norm(dif(:,:,1), 2) + norm(dif(:,:,2), 2) + norm(dif(:,:,3), 2)))));
               weight = double((d_i * d_j - mask_w + 1)  / (0.001 * d_i * d_j * (norm(dif(:,:,1), 2) + norm(dif(:,:,2), 2) + norm(dif(:,:,3), 2))));
               weight = min(1000.0, max(.01, weight));
               weights(i:i+d_i,j:j+d_j,1:dB) = weights(i:i+d_i,j:j+d_j,1:dB) + weight;
               votes(i:i+d_i,j:j+d_j,1:dB) = v_p + (double(B_p) * weight);
           end
        end
    end
    weights(weights==0) = 1;
    votes = votes ./ weights;
    A(repmat(mask, [1,1,dB])) = uint8(votes(repmat(mask, [1,1,dB])));
    
    handles.image = A;
    imshow(handles.image);
    drawnow;
    change = sum(sum(sum(imabsdiff(A, A_orig))));
    disp(change);
    if (change < sum(sum(mask)))
        disp('breaking');
        break;
    end
end
disp('filled.');
A_output = A;


% --- Executes on button press in pushbutton3.
function pushbutton3_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton3 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% We will calculate where the region is and calculate the CNN and do other
% things
A = handles.image;
B = A;

width = 16;
iterations = 5;
k = 1;

mask = createMask(handles.region);
A(:,:,1) = roifill(A(:,:,1), mask);
A(:,:,2) = roifill(A(:,:,2), mask);
A(:,:,3) = roifill(A(:,:,3), mask);

if (isfield(handles, 'constraint'))
    constmask = createMask(handles.constraint);
    constmask = (constmask == 0);
    constmask(mask) = 1;
else 
    constmask = mask;
end

% pause on;
% mask = (mask == 0); %invert mask

% A_output = CSH_level(5, A, B, mask, width, iterations, k);
% A_output = A;
A_output = CSH_level(5, A, B, mask, constmask, width, iterations, k, handles);
% A_output = CSH_level(5, A, B, mask, mask, width, iterations, k, handles);
% CSH_fill(A, B, mask, width, iterations, k, handles);
% handles.image = A_output;
figure;
imshow(A_output);

% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);
if (isfield(handles, 'region'))
    delete(handles.region);
    handles = rmfield(handles, 'region');
end

handles.region = impoly();
guidata(hObject, handles);


% --- Executes on button press in pushbutton5.
function pushbutton5_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton5 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
axes(handles.axes1);
if (isfield(handles, 'constraint'))
    delete(handles.constraint);
    handles = rmfield(handles, 'constraint');
end

handles.constraint = impoly();
% handles.eps = getPosition(handles.constraint);
guidata(hObject, handles);
