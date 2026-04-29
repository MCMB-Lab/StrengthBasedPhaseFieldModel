%% calculate ce 1
close all;clear;clc;
syms sigma_t sigma_c lambda mu Gc varepsilon beta2 beta1 delta_varepsilon
% tensile
k=lambda+2*mu/3;
I1=sigma_t;
J2=sigma_t^2/3;
W=J2/2/mu+I1^2/18/k;
Ce=beta2*sqrt(J2)+beta1*I1+3*Gc*delta_varepsilon/8/varepsilon;
equ1=8*W/3-4*Ce/3-Gc/2/varepsilon==0;
% compressive
kk=lambda+2*mu/3;
II1=-sigma_c;
JJ2=sigma_c^2/3;
WW=JJ2/2/mu+II1^2/18/kk;
CCe=beta2*sqrt(JJ2)+beta1*II1+3*Gc*delta_varepsilon/8/varepsilon;
equ2=8*WW/3-4*CCe/3-Gc/2/varepsilon==0;
% solve
[beta_2,beta_1]=solve(equ1,equ2,beta2,beta1);
latex(beta_1)
latex(beta_2)
s=solve(equ1,equ2,[beta2,beta1]);
latex(s.beta1)
latex(s.beta2)

%% calculate ce 2
close all;clear;clc;
syms sigma_t sigma_c lambda mu Gc varepsilon beta2 beta1 delta_varepsilon
% tensile
k=lambda+2*mu/3;
I1=sigma_t;
J2=sigma_t^2/3;
W=J2/2/mu+I1^2/18/k;
Ce=beta2*sqrt(J2)+beta1*I1+3*Gc*delta_varepsilon/8/varepsilon;
equ1=8*W/3-4*Ce/3-Gc/2/varepsilon==0;
% compressive
kk=lambda+2*mu/3;
II1=-sigma_c;
JJ2=sigma_c^2/3;
WW=JJ2/2/mu+II1^2/18/kk;
CCe=beta2*sqrt(JJ2)+beta1*II1+3*Gc*delta_varepsilon/8/varepsilon+2*(JJ2/2/mu+II1^2/18/kk);
equ2=8*WW/3-4*CCe/3-Gc/2/varepsilon==0;
% solve
[beta_2,beta_1]=solve(equ1,equ2,beta2,beta1);
latex(beta_1)
latex(beta_2)

%% calculate ce 3
close all;clear;clc;
syms sigma_t sigma_c lambda mu Gc varepsilon beta2 beta1 delta_varepsilon nn sigma_h
% tensile
k=lambda+2*mu/3;
I1=sigma_t;
J2=sigma_t^2/3;
W=J2/2/mu+I1^2/18/k;
Ce=beta2*sqrt(J2)+beta1*I1+3*Gc*delta_varepsilon/8/varepsilon;
equ1=8*W/3-4*Ce/3-Gc/2/varepsilon==0;
% compressive
kk=lambda+2*mu/3;
II1=-sigma_c;
JJ2=sigma_c^2/3;
WW=JJ2/2/mu+II1^2/18/kk;
CCe=beta2*sqrt(JJ2)+beta1*II1+3*Gc*delta_varepsilon/8/varepsilon+nn*2*(JJ2/2/mu+II1^2/18/kk);
equ2=8*WW/3-4*CCe/3-Gc/2/varepsilon==0;
% solve
[beta_2,beta_1]=solve(equ1,equ2,beta2,beta1);
latex(beta_1)
latex(beta_2)

syms sigma_t sigma_c lambda mu Gc varepsilon beta_2 beta_1 delta_varepsilon nn sigma_h
% hydrastaatic compression
kk=lambda+2*mu/3;
II1=-3*sigma_h;
JJ2=0;
WW=JJ2/2/mu+II1^2/18/kk;
CCe=beta_2*sqrt(JJ2)+beta_1*II1+3*Gc*delta_varepsilon/8/varepsilon+nn*2*(JJ2/2/mu+II1^2/18/kk);
equh=8*WW/3-4*CCe/3-Gc/2/varepsilon==0;
sig=solve(equh,sigma_h);
latex(sig)

%% classical phase field model
close all;clear;clc;
miu=4300;% shear modulous, [MPa]
k=4400;% bulk modulous, [MPa]
E=9800;% Young's modulous, [MPa]
sigt=27;% tensile strength, [MPa]
sig1=-58:0.5:58;
n=size(sig1,2);
sig2=zeros(2,n);
syms x
for ii=1:n
    eqn=(((sig1(ii)-x)^2+sig1(ii)^2+x^2)/6/miu+(sig1(ii)+x)^2/9/k)/2-sigt^2/E==0;
    y=solve(eqn,x);
    sig2(:,ii)=eval(y);
end
figure
plot(sig1,sig2(1,:))
hold on
plot(sig1,sig2(2,:))
plot([-120,60],[0,0],'k','LineWidth',1)
plot([0,0],[-120,60],'k','LineWidth',1)
plot([-120,60],[-120,60],'--k','LineWidth',1)
axis equal
xlim([-120,60])
ylim([-120,60])
xlabel('$ \sigma_1\ \rm [MPa] $','Interpreter','Latex')
ylabel('$ \sigma_2\ \rm [MPa] $','Interpreter','Latex')


figure
plot(sig1,sig2(1,:),'color',[0.0000,0.4470,0.7410]);
hold on
plot(sig1,sig2(2,:),'color',[0.0000,0.4470,0.7410]);
plot([-60,60],[0,0],'k','LineWidth',1)
plot([0,0],[-60,60],'k','LineWidth',1)
plot([-60,60],[-60,60],'--k','LineWidth',1)
axis equal
xlim([-60,60])
ylim([-60,60])
xlabel('$ \sigma_1\ \rm [MPa] $','Interpreter','Latex')
ylabel('$ \sigma_2\ \rm [MPa] $','Interpreter','Latex')

%% determine the constants for Drucker-Prager strength surface
% close all;clear;clc;
% % sig1=sigt, sig2=-sigc
% syms sigc sigt c1 c2
% solve(sqrt(sigt^2/3)+c1*sigt+c2==0,sqrt(sigc^2/3)+c1*(-sigc)+c2==0,c1,c2)

%% Drucker-Prager strength surface
% close all;clear;clc;
% sigt=27;% tensile strength, [MPa]
% sigc=77;% compressive strength, [MPa]
% c1=(sigc-sigt)/sqrt(3)/(sigc+sigt);
% c2=-2*sigc*sigt/sqrt(3)/(sigc+sigt);
% sig1=-120:1:28;
% n=size(sig1,2);
% sig2=zeros(2,n);
% syms x
% for ii=1:n
% %     eqn=sqrt(sig1(ii)^2+x^2)+c1*(sig1(ii)+x)-c2==0;
%     eqn=sqrt(((sig1(ii)-x)^2+sig1(ii)^2+x^2)/6)+c1*(sig1(ii)+x)+c2==0;
%     y=solve(eqn,x);
%     sig2(:,ii)=eval(y);
% end
% figure
% plot(sig1,sig2(1,:))
% hold on
% plot(sig1,sig2(2,:))
% plot([-120,60],[0,0],'k','LineWidth',1)
% plot([0,0],[-120,60],'k','LineWidth',1)
% plot([-120,60],[-120,60],'--k','LineWidth',1)
% axis equal
% xlim([-120,60])
% ylim([-120,60])
% xlabel('$ \sigma_1\ \rm [MPa] $','Interpreter','Latex')
% ylabel('$ \sigma_2\ \rm [MPa] $','Interpreter','Latex')

%% stress-strain curve
% close all;clear;clc;
% E=25000;
% nu=0.3;
% lmbda=E*nu/(1.0+nu)/(1.0-2.0*nu);
% mu=E/2.0/(1.0+nu);
% k=lmbda+2.0*mu/3.0;
% Gc=0.05;
% sigt=15.0;
% sigc=30.0;
% lIrwin=3.0*Gc*E/8.0/(sigt^2);
% lc=lIrwin/0.92;
% 
% eps1=(0:0.00001:0.002);
% eps2=eps1;
% n=size(eps1,2);
% eps=zeros(3,n);
% eps(1,:)=eps1;
% eps(2,:)=eps1;
% eps(3,:)=0;
% sig=zeros(3,n);
% h=zeros(1,n);
% for ii=1:n
%     sig(:,ii)=E*[1,nu,0;nu,1,0;0,0,1-nu]*eps(:,ii)/(1-nu^2);
%     h(1,ii)=(1+nu)*(sig(1,ii)^2+sig(2,ii)^2)/2/E-nu*(sig(1,ii)+sig(2,ii))^2/2/E;
% end
% 
% d1=zeros(1,n);
% sig1=zeros(3,n);
% d1=1-3*Gc./8./lc./2./h;
% for ii=1:n
%     sig1(:,ii)=(1-d1(ii))^2*E*[1,nu,0;nu,1,0;0,0,1-nu]*eps(:,ii)/(1-nu^2);
% end
% d1new=zeros(1,n);
% sig1new=zeros(3,n);
% for ii=1:n
%     if(d1(ii)<0)
%         d1new(ii)=0;
%     else
%         d1new(ii)=d1(ii);
%     end
%     sig1new(:,ii)=(1-d1new(ii))^2*E*[1,nu,0;nu,1,0;0,0,1-nu]*eps(:,ii)/(1-nu^2);
% end
% figure
% plot(eps(1,:),d1)
% hold on
% plot(eps(1,:),d1new,'--')
% ylim([-1,1])
% figure
% plot(eps(1,:),sig1(1,:))
% hold on
% plot(eps(1,:),sig1new(1,:),'--')
% ylim([-5,20])
% xlabel('$ \varepsilon\ \rm [mm/mm] $','Interpreter','Latex')
% ylabel('$ \sigma\ \rm [MPa] $','Interpreter','Latex')
% 
% lc=lIrwin/5.0;
% delta=0.1;
% beta2=-sqrt(3.0)*(1.0+delta)*(sigc+sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc+sqrt(3.0)*lmbda*(sigc+sigt)/2.0/mu/(3.0*lmbda+2.0*mu)+sqrt(3.0)*(sigc+sigt)/2.0/(3.0*lmbda+2.0*mu);
% beta1=-(1.0+delta)*(sigc-sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc-lmbda*(sigc-sigt)/2.0/mu/(3.0*lmbda+2.0*mu)-(sigc-sigt)/2.0/(3.0*lmbda+2.0*mu);
% beta0=delta*3.0*Gc/8.0/lc;
% J2=zeros(1,n);
% I1=zeros(1,n);
% for ii=1:n
%     J2(1,ii)=((sig(1,ii)-sig(2,ii))^2+(sig(1,ii)-0)^2+(-sig(2,ii)-0)^2)/6;
%     I1(1,ii)=sig(1,ii)+sig(2,ii);
% end
% dsb=zeros(2,n);
% sigsb=zeros(3,n);
% for ii=1:n
%     syms dd
%     if ii==1
%         
%     else
%         equn=2*(1-dd)*h(1,ii)-3*Gc/8/lc-(1-dd)^2*beta2*sqrt(J2(1,ii))-(1-dd)^2*beta1*I1(1,ii)-beta0==0;
%         y=solve(equn,dd);
%         dsb(1,ii)=eval(y(1));
%         dsb(2,ii)=eval(y(2));
%     end
% end
% for ii=1:n
%     sigsb(:,ii)=(1-dsb(1,ii))^2*E*[1,nu,0;nu,1,0;0,0,1-nu]*eps(:,ii)/(1-nu^2);
% end
% 
% dsbnew=zeros(2,n);
% sigsbnew=zeros(3,n);
% for ii=1:n
%     if(dsb(1,ii)<0)
%         dsbnew(1,ii)=0;
%         dsbnew(2,ii)=0;
%     elseif(dsb(1,ii)>1.0)
%         dsbnew(1,ii)=1;
%         dsbnew(2,ii)=1;
%     else
%         dsbnew(1,ii)=dsb(1,ii);
%         dsbnew(2,ii)=dsb(2,ii);
%     end
%     sigsbnew(:,ii)=(1-dsbnew(1,ii))^2*E*[1,nu,0;nu,1,0;0,0,1-nu]*eps(:,ii)/(1-nu^2);
% end
% 
% figure
% plot(eps(1,:),dsb(1,:))
% hold on
% plot(eps(1,:),dsbnew(1,:),'--')
% ylim([-1,1])
% figure
% plot(eps(1,2:end),sigsb(1,2:end))
% hold on
% plot(eps(1,2:end),sigsbnew(1,2:end),'--')
% ylim([-5,20])
% xlabel('$ \varepsilon\ \rm [mm/mm] $','Interpreter','Latex')
% ylabel('$ \sigma\ \rm [MPa] $','Interpreter','Latex')

%% derive strength surface
% close all;clear;clc;
% % AT4
% E=25000;
% nu=0.3;
% lmbda=E*nu/(1.0+nu)/(1.0-2.0*nu);
% mu=E/2.0/(1.0+nu);
% k=lmbda+2.0*mu/3.0;
% Gc=0.05;
% sigt=5.0;
% sigc=28.0;
% delta=10;
% lIrwin=3.0*Gc*E/8.0/(sigt^2);
% lc=lIrwin/50.0;
% beta2=-sqrt(3.0)*(1.0+delta)*(sigc+sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc+sqrt(3.0)*lmbda*(sigc+sigt)/2.0/mu/(3.0*lmbda+2.0*mu)+sqrt(3.0)*(sigc+sigt)/2.0/(3.0*lmbda+2.0*mu);
% beta1=-(1.0+delta)*(sigc-sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc-lmbda*(sigc-sigt)/2.0/mu/(3.0*lmbda+2.0*mu)-(sigc-sigt)/2.0/(3.0*lmbda+2.0*mu);
% beta0=delta*3.0*Gc/8.0/lc;
% sig1=-60:1:5;
% n=size(sig1,2);
% sig2=zeros(2,n);
% for ii=1:n
%     syms sig2pre
%     J2=((sig1(ii)-sig2pre)^2+(sig1(ii)-0)^2+(sig2pre-0)^2)/6;
%     I1=sig1(ii)+sig2pre;
%     h=2*(J2/mu+I1^2/9/k)/2;
%     equn=2*h-3*Gc/8/lc-beta2*sqrt(J2)-beta1*I1-beta0==0;
%     y=vpasolve(equn,sig2pre,5);
%     sig2(1,ii)=eval(y(1));
% %     sig2(2,ii)=eval(y(2));
% end
% figure
% plot(sig1,sig2(1,:))
% hold on
% plot(sig2(1,:),sig1)
% plot([-40,10],[0,0])
% plot([0,0],[-40,10])
% axis equal
% 
% % AT5
% E=25000;
% nu=0.3;
% lmbda=E*nu/(1.0+nu)/(1.0-2.0*nu);
% mu=E/2.0/(1.0+nu);
% k=lmbda+2.0*mu/3.0;
% Gc=0.05;
% sigt=5.0;
% sigc=28.0;
% delta=10;
% lIrwin=3.0*Gc*E/8.0/(sigt^2);
% lc=lIrwin/20.0;
% beta2=-sqrt(3.0)*(1.0+delta)*(sigc+sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc+sigt/2.0/sqrt(3.0)/(3.0*lmbda+2.0*mu)+sigt/2.0/sqrt(3.0)/mu;
% beta1=-(1.0+delta)*(sigc-sigt)*3.0*Gc/2.0/sigc/sigt/8.0/lc+sigt/6.0/(3.0*lmbda+2.0*mu)+sigt/6.0/mu;
% beta0=delta*3.0*Gc/8.0/lc;
% % sig1=-60:1:5;
% n=size(sig1,2);
% sig2at5=zeros(2,n);
% for ii=1:n
%     syms sig2pre
%     J2=((sig1(ii)-sig2pre)^2+(sig1(ii)-0)^2+(sig2pre-0)^2)/6;
%     I1=sig1(ii)+sig2pre;
%     h=2*(J2/mu+I1^2/9/k)/2;
%     equn=2*h-3*Gc/8/lc-beta2*sqrt(J2)-beta1*I1-beta0-2*h==0;
%     y=vpasolve(equn,sig2pre,5);
%     sig2at5(1,ii)=eval(y(1));
% %     sig2(2,ii)=eval(y(2));
% end
% figure
% plot(sig1,sig2at5(1,:))
% hold on
% plot(sig2at5(1,:),sig1)
% plot([-40,10],[0,0])
% plot([0,0],[-40,10])
% axis equal
% 
% 
% figure
% h1=plot(sig1,sig2(1,:),'color',[0.0000,0.4470,0.7410]);
% hold on
% plot(sig2(1,:),sig1,'color',[0.0000,0.4470,0.7410])
% h2=plot(sig1,sig2at5(1,:),'--','color',[0.8500,0.3250,0.0980]);
% plot(sig2at5(1,:),sig1,'--','color',[0.8500,0.3250,0.0980])
% plot([-60,10],[0,0],'k','linewidth',0.5)
% plot([0,0],[-60,10],'k','linewidth',0.5)
% axis equal
% legend([h1,h2],'$ \rm original\ strength{-}based\ PFM $','$ \rm stiffer\ strength{-}based\ PFM $','FontSize',15,'Location','SouthWest','Interpreter','Latex');
% xlabel('$ \sigma_1\ \rm [MPa] $','Interpreter','Latex')
% ylabel('$ \sigma_2\ \rm [MPa] $','Interpreter','Latex')

