from django.shortcuts import render, HttpResponse
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel

from django.core.files.storage import FileSystemStorage

# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid=request.POST.get("loginid")
        password=request.POST.get("pswd")
        print(loginid)
        print(password)
        try:
            check=UserRegistrationModel.objects.get(loginid=loginid,password=password)
            status=check.status
            if status=="activated":
                request.session['id']=check.id
                request.session['loginid']=check.loginid
                request.session['password']=check.password
                request.session['email']=check.email
                return render(request,'users/UserHome.html')
            else:
                messages.success(request,"your account not activated")
            return render(request,"UserLogin.html")
        except Exception as e:
            print('=======>',e)
        messages.success(request,'invalid details')
    return render(request,'UserLogin.html')
    
def UserHome(request):
    return render(request,"users/UserHome.html",{})


def Chest(request):
    if request.method == 'POST':
        import cv2
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        from .utility.predictChest import start_process
        results_class = start_process(filename)
        print(results_class)
        import os
        from django.conf import settings
        img_path = os.path.join(settings.MEDIA_ROOT,filename)
        image = cv2.imread(img_path, 0)
        print('-'*100)
        print(image.shape)
  
        # Window name in which image is displayed
        window_name = results_class
        dim = (512,512)
        # Using cv2.imshow() method
        # Displaying the image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow(window_name, resized)
        
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        
        # closing all open windows
        cv2.destroyAllWindows()
        return render(request,'users/chest_predict.html',{'results_class':results_class,'path':uploaded_file_url})
    else:
        return render(request,'users/chest_predict.html')


def Mammography(request):
    if request.method == 'POST':
        import cv2
        import os
        from django.conf import settings
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        from .utility.predictMammography import start_process
        results_class = start_process(filename)
        print(results_class)
        img_path = os.path.join(settings.MEDIA_ROOT,filename)
        image = cv2.imread(img_path, 0)
        print('-'*100)
        print(image.shape)
  
        # Window name in which image is displayed
        window_name = results_class
        dim = (512,512)
        # Using cv2.imshow() method
        # Displaying the image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow(window_name, resized)
        
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        
        # closing all open windows
        cv2.destroyAllWindows()
        return render(request,'users/mammography_predict.html',{'results_class':results_class,'path':uploaded_file_url})
    return render(request,'users/mammography_predict.html')


def MriStroke(request):
    if request.method == 'POST':
        import cv2
        import os
        from django.conf import settings
        myfile = request.FILES['file']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
        from .utility.predictMriStroke import start_process
        res = start_process(filename)
        if res > 0.5:
            results_class = 'Tumor Detected'
        else:
            results_class = 'No Tumor Detected'
          
        img_path = os.path.join(settings.MEDIA_ROOT,filename)
        image = cv2.imread(img_path, 0)
        print('-'*100)
        print(image.shape)
  
        # Window name in which image is displayed
        window_name = results_class
        dim = (512,512)
        # Using cv2.imshow() method
        # Displaying the image
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        cv2.imshow(window_name, resized)
        
        # waits for user to press any key
        # (this is necessary to avoid Python kernel form crashing)
        cv2.waitKey(0)
        
        # closing all open windows
        cv2.destroyAllWindows()  
        
        return render(request,'users/mri_stroke_predict.html',{'results_class':results_class,'res':res})
    else:
        return render(request,'users/mri_stroke_predict.html')
        
    