import time
import numpy as np
import mujoco
import mujoco.viewer
import pickle
from simple_pid import PID

def save_data(filename, positions, velocities):
    data = {'positions': positions, 'velocities': velocities}
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
        """Bu fonksiyon, drone'un konum ve hız verilerini alır ve bunları bir dosyaya pickle formatında kaydeder."""

def pid_to_thrust(input: np.array):  #inputu np.array şeklinde alıyorum.
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [-0.25, 0.25, 0.25, -0.25],
      [0.25, 0.25, -0.25, -0.25],
      [-0.25, 0.25, -0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

"""Bu matris, PID kontrolcü çıktılarını (örneğin, dönüş açıları) dört motorun her birinin üreteceği torklara (thrust) dönüştürmek için kullanılır.
3x4'lük bu matrisin satırları, her bir motorun etkisini belirler.
İlk satırda (x eksenindeki kontrol) motorların roll yönündeki etkileri, ikinci satırda (y eksenindeki kontrol) pitch yönündeki etkiler ve üçüncü satırda (z eksenindeki kontrol) yaw yönündeki etkiler bulunur."""

"""input parametresi bir np.array'dir ve genellikle w olarak belirtilmiştir. Bu input, PID kontrolcüsünden gelen çıkışları temsil eder.
c_to_F * input ifadesi, her bir kontrol girdisinin (örneğin, x, y, z yönünde) belirli motorlar üzerindeki etkisini hesaplar.
Sonra np.dot(...) ifadesi, bu etkileri motorlara yönlendirmek için dot (skaler çarpma) işlemi yapar. Sonuç, her bir motorun gerekli tork seviyesini verir."""

def outer_pid_to_thrust(input: np.array):
  """ Maps controller output to manipulated variable.

  Args:
      input (np.array): w € [3x1]

  Returns:
      np.array: [3x4]
  """
  c_to_F =np.array([
      [0.25, 0.25, -0.25, -0.25],
      [0.25, -0.25, -0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25]
  ]).transpose()

  return np.dot((c_to_F*input),np.array([1,1,1]))

"""Bu matris de, PID çıkışlarını dört motorun torklarına dönüştürmek için kullanılır.
 Buradaki fark, her motorun x, y, z eksenlerindeki etkilerinin farklı şekilde paylaştırılmış olmasıdır.
İlk satır x yönündeki torkları, ikinci satır y yönündeki torkları ve üçüncü satır z yönündeki torkları belirler."""

class PDController:
  def __init__(self, kp, kd, setpoint):
    self.kp = kp
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.kd * derivative)
    self.prev_error = error
    return output

  """Proportional (P) Bileşeni:
Hata (error): Kontrol edilen değer (örneğin, dronun mevcut konumu veya hızı) ile hedeflenen değer (setpoint) arasındaki farktır.
Proportional terimi, bu hataya orantılı bir çıktı üretir. Yani, hata ne kadar büyükse, kontrol çıkışı da o kadar büyük olur.
Derivative (D) Bileşeni:
Derivative, hata değişiminin hızını (hata türevini) dikkate alır. Hata değişiyor mu? Hata hızla artıyor mu, yoksa yavaşlıyor mu?
Bu, sistemi daha kararlı hale getirmek için, hatadaki hızlı değişimlere tepki vermek amacıyla kullanılır. Eğer hata hızlıca değişiyorsa, sistem bu değişime karşılık verir."""

"""kp: Proportional (P) katsayısı, hatayı kontrol etmek için kullanılır.
kd: Derivative (D) katsayısı, hata değişim hızını kontrol etmek için kullanılır.
setpoint: Hedeflenen değer (örneğin, istenilen sıcaklık, hız, konum vb.).
prev_error: Önceki hatayı saklamak için kullanılır. Bu, türev hesaplaması için gereklidir."""

"""compute(self, measured_value):
Bu fonksiyon, sistemin kontrol edilen değerini (measured_value) alır ve hedef değerle (setpoint) karşılaştırır.
error: Hedef (setpoint) ile ölçülen değer (measured_value) arasındaki farktır.
derivative: Hata değişimi (türev) hesaplanır. Bu, önceki hata ile şu anki hata arasındaki farktır.
Kontrol çıktısı (output): Bu, orantılı bileşen (P) ve türev bileşeninin (D) toplamıdır. kp ve kd katsayıları, her bileşenin ne kadar etkili olduğunu belirler.
Son olarak, önceki hata (prev_error) güncellenir, böylece bir sonraki hesaplama için doğru türev değerini alabilirsin."""

class PIDController:  #açıklamalarını aşağıya yazıyorum.
  def __init__(self, kp, ki, kd, setpoint):
    self.kp = kp
    self.ki = ki
    self.kd = kd
    self.setpoint = setpoint
    self.prev_error = 0
    self.integral = 0

  def compute(self, measured_value):
    error = self.setpoint - measured_value
    self.integral += error
    derivative = error - self.prev_error
    output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
    self.prev_error = error
    return output

"""Proportional (P) Bileşeni:
Hata (error): Hedeflenen değer (setpoint) ile ölçülen değer (measured_value) arasındaki farktır.
Proportional terimi, bu hatayı dikkate alır ve hataya orantılı bir çıktı üretir. Hata ne kadar büyükse, kontrol çıkışı o kadar büyük olur.
Integral (I) Bileşeni:
Integral, hataların zamanla birikmesini ve bu birikimlerin etkisini dikkate alır.
Sürekli hata durumunda (örneğin, bir sistemin uzun süre küçük bir hata ile çalışması) integral bileşeni, bu küçük hataları toplar ve sistemin hedefe ulaşmasını hızlandırmak için bir düzeltme sinyali üretir.
Derivative (D) Bileşeni:
Derivative, hata değişiminin hızını (türevini) dikkate alır.
Bu bileşen, hatanın hızla değişmesini göz önünde bulundurur ve hatadaki ani değişimlere tepki verir. Derivative terimi, sistemi daha kararlı hale getirebilir."""

"""__init__(self, kp, ki, kd, setpoint):
Bu, sınıfın kurucusudur. Burada üç kontrol parametresi (kp, ki, kd) ve bir hedef değer (setpoint) belirtilir.
kp: Proportional (P) katsayısı, hatayı kontrol etmek için kullanılır.
ki: Integral (I) katsayısı, hataların birikmesini kontrol etmek için kullanılır.
kd: Derivative (D) katsayısı, hata değişim hızını kontrol etmek için kullanılır.
setpoint: Hedeflenen değer (örneğin, dronun hedef yüksekliği veya sıcaklık).
prev_error: Önceki hatayı saklar, bu değer türev hesaplamasında kullanılır.
integral: Hata birikimini tutar, bu değer zamanla birikir."""

"""compute(self, measured_value):
Bu fonksiyon, kontrol edilen değeri (measured_value) alır ve hedef değeri (setpoint) ile karşılaştırır.
error: Hedef (setpoint) ile ölçülen değer (measured_value) arasındaki farktır.
integral: Hata birikimini tutar. Bu, integral bileşeninin etkisini temsil eder.
derivative: Hata değişimi (türev) hesaplanır, yani şu anki hata ile önceki hata arasındaki farktır.
Kontrol çıktısı (output): Bu, PID kontrolörü tarafından üretilen toplam çıkıştır. Proportional, Integral ve Derivative bileşenlerinin etkileri toplanır ve bir kontrol sinyali oluşturulur.
Son olarak, önceki hata (prev_error) güncellenir ve bir sonraki hesaplama için doğru türev değeri alınılır."""

#NEDEN HEM PD HEM DE PID KULLANMIŞIM ?????????******************************
"""PD genellikle hızlı tepki sağlamak ve sistemdeki geçici değişimlere hızla uyum sağlamak için tercih edilir.
PID, uzun süreli hataların giderilmesi ve daha dengeli bir kontrol sağlanması için ek olarak kullanılır."""


class dummyPlanner:       #Bu CLASS, bir drone için bir yol planlayıcısı (path planner) sınıfı oluşturur.
  # dummyPlanner sınıfı, bir hedef noktaya uçuş yolunu ve hız vektörünü hesaplamak için kullanılır.
  """Generate Path from 1 point directly to another"""

  def __init__(self, target, vel_limit = 2) -> None:
    # TODO: MPC
    self.target = target  #target: Hedef noktasını belirtir (örneğin, bir x, y, z koordinatı)
    self.vel_limit = vel_limit  #vel_limit: Drone'un maksimum hızını belirler (varsayılan olarak 2 birim).
    # setpoint target location, controller output: desired velocity.
    self.pid_x = PID(2, 0.15, 1.5, setpoint = self.target[0],
                output_limits = (-vel_limit, vel_limit),)
    self.pid_y = PID(2, 0.15, 1.5, setpoint = self.target[1],
                output_limits = (-vel_limit, vel_limit))
    """Bu metod, iki adet PID kontrolörü oluşturur: biri x ekseninde, diğeri ise y ekseninde drone'un hareketini kontrol eder.
         PID kontrolörü, drone'un hedefe doğru nasıl hareket edeceğini belirler."""
  
  def __call__(self, loc: np.array):  #Bu metod, sınıf örneğini çağırarak drone'un yön ve hız komutlarını hesaplar.
    #loc: Mevcut konum.
    """Calls planner at timestep to update cmd_vel"""
    velocites = np.array([0,0,0])
    velocites[0] = self.pid_x(loc[0])
    velocites[1] = self.pid_y(loc[1])
    return velocites
    """Bu function, PID kontrolörü tarafından verilen hız komutlarını alır ve x ve y eksenindeki hızları (velocities) döndürür.
    aynı zamanda bu function drone'un hareket etmesi için gerekli olan hız bilgilerini sağlar."""
  def get_velocities(self,loc: np.array, target: np.array,
                     time_to_target: float = None,
                     flight_speed: float = 0.5) -> np.array:
    """Compute

    Args:
        loc (np.array): Current location in world coordinates.
        target (np.array): Desired location in world coordinates
        time_to_target (float): If set, adpats length of velocity vector.

    Returns:
        np.array: returns velocity vector in world coordinates.
    """

    direction = target - loc
    distance = np.linalg.norm(direction)
    # maps drone velocities to one.
    if distance > 1:
        velocities = flight_speed * direction / distance

    else:
        velocities =  direction * distance

    return velocities
    """Hız vektörünü elde etmek için hedef ile şu anki konum arasındaki yön (direction) ve mesafe (distance) hesaplanır.
Eğer mesafe 1 birimden büyükse, belirli bir hızda (flight_speed) hedefe doğru hareket edilir.
Eğer mesafe 1 birimden küçükse, hareket hızı mesafe ile orantılı olarak küçültülür."""

  def get_alt_setpoint(self, loc: np.array) -> float:
    """"loc: Mevcut konum (yükseklik bilgisi de dahil)."""

    target = self.target
    distance = target[2] - loc[2]
    
    # maps drone velocities to one.
    if distance > 0.5:
        time_sample = 1/4
        time_to_target =  distance / self.vel_limit
        number_steps = int(time_to_target/time_sample)
        # compute distance for next update
        delta_alt = distance / number_steps

        # 2 times for smoothing
        alt_set = loc[2] + 2 * delta_alt
    
    else:
        alt_set = target[2]

    return alt_set
  """Hedef yüksekliğe ulaşmak için, drone'a yeni bir yükseklik set noktası (altitude setpoint) atanır. Eğer yükseklik farkı çok küçükse, drone hemen hedef yüksekliğine ayarlanır.
Eğer fark büyükse, bu farkı bölerek daha düzgün bir geçiş için adım adım bir hız profili hesaplanır."""

  def update_target(self, target):
    """Update targets"""
    self.target = target  # Yeni hedef koordinatı (x, y, z).
    # setpoint target location, controller output: desired velocity.
    self.pid_x.setpoint = self.target[0]
    self.pid_y.setpoint = self.target[1]
  """Bu metod, yeni hedefe yönelik PID kontrol sistemini günceller."""

class dummySensor:
  """Dummy sensor data. So the control code remains intact."""
  def __init__(self, d):
    self.position = d.qpos
    self.velocity = d.qvel
    self.acceleration = d.qacc

  def get_position(self):
    return self.position #Bu metod, sensörden alınan konum verisini döndürür.
    # self.position, simülasyon ortamındaki aracın (örneğin, bir drone) mevcut konumunu temsil eder.
  
  def get_velocity(self):
    return self.velocity #Bu metod, sensörden alınan hız verisini döndürür.
    # self.velocity, aracın simülasyondaki hızını temsil eder.
  
  def get_acceleration(self):
    return self.acceleration #Bu metod, sensörden alınan ivme verisini döndürür.
    # self.acceleration, aracın simülasyondaki ivme (yani, hızındaki değişim oranı) bilgisini temsil eder.
"""d: Bu parametre genellikle bir simülasyon ortamındaki sistem durumu veya simülasyon verisidir.
 d.qpos, d.qvel ve d.qacc gibi alanlar genellikle bir aracın konum, hız ve ivme gibi dinamik özelliklerini temsil eder.
qpos: Konum verisi (örneğin, drone'un uzaydaki pozisyonu).
qvel: Hız verisi (örneğin, drone'un hareket hızı).
qacc: İvme verisi (örneğin, drone'un hız değişim oranı).
Bu veriler sınıfın içinde saklanır ve gerçek bir sensör gibi çalışacak şekilde alınabilir."""

"""dummySensor sınıfı, gerçek sensör verilerini taklit eder ve simülasyonlarda kontrol algoritmalarının test edilmesine olanak tanır.
 Bu sınıf sayesinde sensör verileri olmadığı durumlarda, kontrol kodu bozulmaz ve doğru çalışmaya devam eder."""
class drone:
  """Simple drone classe."""
  def __init__(self, target=np.array((0,0,0))):
    self.m = mujoco.MjModel.from_xml_path('mujoco_menagerie-main/skydio_x2/scene.xml')
    # self.m: Mujoco modelini başlatır.
    # Bu, drone'un fiziksel modelini tanımlar ve simülatördeki drone'un fiziksel özelliklerini temsil eder.
    self.d = mujoco.MjData(self.m)
    #self.d: MjData nesnesi, simülasyonun durum bilgilerini tutar (örneğin, hız, konum).

    self.planner = dummyPlanner(target=target) #self.planner: dummyPlanner nesnesi, hedefe ulaşmak için gerekli yolu hesaplar.
    self.sensor = dummySensor(self.d) #self.sensor: dummySensor nesnesi, drone'un mevcut sensör verilerini sağlar (örneğin, konum, hız, ivme).

    # instantiate controllers

    # inner control to stabalize inflight dynamics
    self.pid_alt = PID(5.50844,0.57871, 1.2,setpoint=0,) # PIDController(0.050844,0.000017871, 0, 0) # thrust
    self.pid_roll = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) ) #PID(11.0791,2.5263, 0.10513,setpoint=0, output_limits = (-1,1) )
    self.pid_pitch = PID(2.6785,0.56871, 1.2508, setpoint=0, output_limits = (-1,1) )
    self.pid_yaw =  PID(0.54, 0, 5.358333, setpoint=1, output_limits = (-3,3) )# PID(0.11046, 0.0, 15.8333, setpoint=1, output_limits = (-2,2) )

    """self.pid_alt: Yükseklik kontrolü için PID kontrolörü.
    self.pid_roll, self.pid_pitch, self.pid_yaw: Dönme hareketlerini (roll, pitch, yaw) kontrol etmek için PID kontrolörleri."""
    # outer control loops
    self.pid_v_x = PID(0.1, 0.003, 0.02, setpoint = 0,
                output_limits = (-0.1, 0.1))
    self.pid_v_y = PID(0.1, 0.003, 0.02, setpoint = 0,
                  output_limits = (-0.1, 0.1))
    """self.pid_v_x, self.pid_v_y: Hedefe ulaşmak için drone'un x ve y eksenlerindeki hızlarını kontrol etmek için PID kontrolörleri."""
  def update_outer_conrol(self):
    """Updates outer control loop for trajectory planning"""
    v = self.sensor.get_velocity()  #v: Drone'un mevcut hızını alır.
    location = self.sensor.get_position()[:3]  #location: Drone'un mevcut konumunu alır.

    # Compute velocites to target
    velocites = self.planner(loc=location)   #velocites: dummyPlanner kullanılarak hedefe ulaşmak için gereken hızları hesaplar.
    
    # In this example the altitude is directly controlled by a PID
    self.pid_alt.setpoint = self.planner.get_alt_setpoint(location) #self.pid_alt.setpoint: Hedef yükseklik için PID kontrolörünü günceller.
    self.pid_v_x.setpoint = velocites[0]
    self.pid_v_y.setpoint = velocites[1]
#self.pid_v_x.setpoint, self.pid_v_y.setpoint: Hedef doğrultusunda drone'un x ve y eksenlerindeki hız hedeflerini ayarlar.
    # Compute angles and set inner controllers accordingly
    angle_pitch = self.pid_v_x(v[0])
    angle_roll = - self.pid_v_y(v[1])
#angle_pitch ve angle_roll: X ve Y yönlerindeki hızlara göre pitch (eğilme) ve roll (dönme) açıları hesaplanır.
    self.pid_pitch.setpoint= angle_pitch
    self.pid_roll.setpoint = angle_roll
#self.pid_pitch.setpoint ve self.pid_roll.setpoint: Hesaplanan açıları PID kontrolörlerine set eder.

  def update_inner_control(self):
    """Upates inner control loop and sets actuators to stabilize flight
    dynamics"""
    alt = self.sensor.get_position()[2] #alt: Drone'un mevcut yüksekliği alınır.
    angles = self.sensor.get_position()[3:] # roll, yaw, pitch angles: Drone'un roll, yaw ve pitch açıları alınır.
    
    # apply PID
    cmd_thrust = self.pid_alt(alt) + 3.2495
    cmd_roll = - self.pid_roll(angles[1])
    cmd_pitch = self.pid_pitch(angles[2])
    cmd_yaw = - self.pid_yaw(angles[0])
#cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw: PID kontrolörleri kullanılarak drone'un uçuş komutları (itme gücü, roll, pitch ve yaw açıları) hesaplanır.
    #transfer to motor control
    out = self.compute_motor_control(cmd_thrust, cmd_roll, cmd_pitch, cmd_yaw) #self.compute_motor_control(): Motorlara gönderilmek üzere kontrol sinyalleri hesaplanır.
    self.d.ctrl[:4] = out #self.d.ctrl[:4]: Hesaplanan motor komutları drone'a gönderilir.

  #  as the drone is underactuated we set
  def compute_motor_control(self, thrust, roll, pitch, yaw):
    motor_control = [
      thrust + roll + pitch - yaw,
      thrust - roll + pitch + yaw,
      thrust - roll -  pitch - yaw,
      thrust + roll - pitch + yaw
    ]
    return motor_control
"""Bu metod, PID kontrol döngüsünden elde edilen komutları motor kontrolüne dönüştürür. Drone'nun dört motoru için gerekli itme, roll, pitch ve yaw komutlarını birleştirir.

Dört motor için hesaplanan motor komutları şunlardır:

Motor 1: thrust + roll + pitch - yaw
Motor 2: thrust - roll + pitch + yaw
Motor 3: thrust - roll - pitch - yaw
Motor 4: thrust + roll - pitch + yaw
Bu formüller, drone'un motorlarının hızlarını ayarlayarak drone'un istikrarını ve yönelimini sağlar."""


# -------------------------- Initialization ----------------------------------
my_drone = drone(target=np.array((0,0,1))) #my_drone: drone sınıfından bir nesne oluşturuluyor.
# Bu nesne başlangıçta (0, 0, 1) koordinatlarına doğru uçacak şekilde ayarlanmıştır.

with mujoco.viewer.launch_passive(my_drone.m, my_drone.d) as viewer: #mujoco.viewer.launch_passive(): Drone'un fiziksel modelini ve verilerini simüle ederken bir görselleştirme penceresi açılır.
  # Bu pencere, drone'un simülasyonunu görsel olarak izlemeyi sağlar.
#viewer: Görselleştirici, simülasyonun görsel çıktısını sağlayan bir arayüzdür.
  time.sleep(5)
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  step = 1

  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()
    
    # flight program
    if time.time()- start > 2:
      my_drone.planner.update_target(np.array((1,1,1)))

    if time.time()- start > 10:
      my_drone.planner.update_target(np.array((-1,1,2)))

    if time.time()- start > 18:
      my_drone.planner.update_target(np.array((-1,-1,0.5)))
    """update_target(): Her 2 saniyede bir yeni bir hedef atanır.
     Drone sırasıyla (1, 1, 1), (-1, 1, 2), (-1, -1, 0.5) gibi yeni hedeflere doğru yönlendirilir."""
    # outer control loop
    if step % 20 == 0:
     my_drone.update_outer_conrol()
    """Dış kontrol döngüsü, her 20 adımda bir çalıştırılır ve drone'un hedefe yönelmesini sağlamak için gerekli hız ve açı komutları hesaplanır."""
    # Inner control loop
    my_drone.update_inner_control()
    """İç kontrol döngüsü, drone'un uçuş dinamiklerini stabilize eder ve PID kontrolörleri kullanarak drone'un yüksekliği, roll, pitch ve yaw açılarını kontrol eder."""
    mujoco.mj_step(my_drone.m, my_drone.d)
    """mujoco.mj_step(): Simülasyondaki bir adımı atar. Bu adımda drone'un hareketi, fiziksel simülasyon dinamiklerine göre güncellenir."""
    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(my_drone.d.time % 2)
    #Bu kısım, görselleştiricinin görsel seçeneklerini değiştirir. Örneğin, her iki saniyede bir simülasyonda temas noktalarını gösterir veya gizler.
    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync() #Bu komut, simülasyonun durumunu günceller ve GUI ile senkronize eder.
    
    # Increment to time slower outer control loop
    step += 1
    
    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = my_drone.m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)
#Bu kısım, simülasyonda bir sonraki adım için bekleme süresini ayarlar. Bu, simülasyonun zamanlamasına göre drone'un kontrol döngülerini uyumlu hale getirir.
