"""Install a project-local Java/Gradle/Android SDK toolchain, without Unity."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
BUILD=ROOT/'.build-tools/android'
CACHE=BUILD/'downloads'


def download(url,name,sha=None):
    CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/name
    temporary=path.with_suffix('.download')
    if path.is_file() and sha and hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
        path.replace(temporary)
    if not path.is_file():
        print('Downloading',name,flush=True)
        for attempt in range(5):
            offset=temporary.stat().st_size if temporary.exists() else 0
            headers={'User-Agent':'OurSekai-build-setup'}
            if offset:headers['Range']=f'bytes={offset}-'
            request=urllib.request.Request(url,headers=headers)
            try:
                with urllib.request.urlopen(request,timeout=120) as source:
                    mode='ab' if source.status==206 and offset else 'wb'
                    expected=int(source.headers.get('Content-Length','0')); received=0
                    with temporary.open(mode) as output:
                        while True:
                            block=source.read(1024*1024)
                            if not block:break
                            output.write(block); received+=len(block)
                if expected and received!=expected:raise IOError('Incomplete download')
                if sha and hashlib.sha256(temporary.read_bytes()).hexdigest()!=sha:
                    raise IOError('Incomplete or invalid checksum')
                with zipfile.ZipFile(temporary) as check:
                    if check.testzip():raise IOError('Archive integrity failed')
                os.replace(temporary,path); break
            except Exception:
                if attempt==4:raise
                time.sleep(2)
    if sha and hashlib.sha256(path.read_bytes()).hexdigest()!=sha:raise ValueError('Checksum failed: '+name)
    return path


def extract(archive,destination):
    destination=Path(destination).resolve()
    destination.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            path=(destination/name).resolve()
            if destination not in path.parents:raise ValueError('Archive path escapes toolchain directory')
        source.extractall(destination)


def main():
    BUILD.mkdir(parents=True,exist_ok=True)
    java=next((BUILD/'jdk').glob('*/bin/java.exe'),None)
    if not java:
        request=urllib.request.Request('https://api.adoptium.net/v3/assets/latest/17/hotspot?architecture=x64&image_type=jdk&os=windows',headers={'User-Agent':'OurSekai-build-setup'})
        with urllib.request.urlopen(request,timeout=60) as response:asset=json.load(response)[0]['binary']['package']
        archive=download(asset['link'],asset['name'],asset['checksum'])
        extract(archive,BUILD/'jdk')
        java=next((BUILD/'jdk').glob('*/bin/java.exe'))
    jdk=java.parent.parent
    gradle=BUILD/'gradle/gradle-8.9/bin/gradle.bat'
    python=BUILD/'python310/tools/python.exe'
    sdk=BUILD/'sdk'
    # Pin the traditional CLI: newer bootstrap versions download a separate CLI.
    sdkmanager=sdk/'cmdline-tools/19.0/bin/sdkmanager.bat'
    tasks=[]
    if not gradle.exists():tasks.append(('gradle','https://services.gradle.org/distributions/gradle-8.9-bin.zip','gradle-8.9-bin.zip'))
    if not python.exists():tasks.append(('python','https://www.nuget.org/api/v2/package/python/3.10.11','python-3.10.11.zip'))
    if not sdkmanager.exists():
        with urllib.request.urlopen('https://dl.google.com/android/repository/repository2-1.xml',timeout=60) as response:repository=ET.parse(response).getroot()
        packages=[p for p in repository if p.tag.endswith('remotePackage') and p.attrib.get('path','').startswith('cmdline-tools;')]
        packages=[p for p in packages if p.find('channelRef') is not None and p.find('channelRef').attrib.get('ref')=='channel-0']
        package=next(p for p in packages if p.attrib['path']=='cmdline-tools;19.0')
        for archive in package.findall('archives/archive'):
            if archive.findtext('host-os')=='windows':
                tasks.append(('sdk','https://dl.google.com/android/repository/'+archive.findtext('complete/url'),'android-cmdline-tools-19.zip'))
                break
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(lambda task:(task[0],download(task[1],task[2])),tasks))
    for kind,archive in results:
        if kind=='gradle':extract(archive,BUILD/'gradle')
        elif kind=='python':extract(archive,BUILD/'python310')
        else:
            unpack=BUILD/'sdk-unpack-19'
            extract(archive,unpack)
            sdkmanager.parent.parent.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(unpack/'cmdline-tools'),str(sdk/'cmdline-tools/19.0'))
    env=os.environ.copy()
    env['JAVA_HOME']=str(jdk); env['ANDROID_HOME']=str(sdk)
    env['ANDROID_USER_HOME']=str(BUILD/'android-user')
    env['PATH']=str(jdk/'bin')+os.pathsep+env.get('PATH','')
    log=ROOT/'artifacts/android-sdk-setup.log'
    log.parent.mkdir(exist_ok=True)
    with log.open('w',encoding='utf-8') as output:
        subprocess.run([str(sdkmanager),'--sdk_root='+str(sdk),'--licenses'],input='y\n'*100,text=True,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(sdkmanager),'--sdk_root='+str(sdk),'platforms;android-35','build-tools;35.0.0','platform-tools'],env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    config=dict(java_home=str(jdk),android_home=str(sdk),gradle=str(gradle),build_python=str(python))
    (BUILD/'build-env.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Project-local Android toolchain configured',flush=True)


if __name__=='__main__':main()
