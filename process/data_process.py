from common import *

## https://www.kaggle.com/c/quickdraw-doodle-recognition/data
## https://www.kaggle.com/inversion/getting-started-viewing-quick-draw-doodles-etc
## https://github.com/googlecreativelab/quickdraw-dataset


#DATA_DIR = '/root/share/project/kaggle/google_doodle/data'



CLASS_NAME=\
['The_Eiffel_Tower', 'The_Great_Wall_of_China', 'The_Mona_Lisa', 'airplane', 'alarm_clock', 'ambulance', 'angel',
 'animal_migration', 'ant', 'anvil', 'apple', 'arm', 'asparagus', 'axe', 'backpack', 'banana', 'bandage', 'barn',
 'baseball', 'baseball_bat', 'basket', 'basketball', 'bat', 'bathtub', 'beach', 'bear', 'beard', 'bed', 'bee',
 'belt', 'bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake', 'blackberry', 'blueberry', 'book',
 'boomerang', 'bottlecap', 'bowtie', 'bracelet', 'brain', 'bread', 'bridge', 'broccoli', 'broom',
 'bucket', 'bulldozer', 'bus', 'bush', 'butterfly', 'cactus', 'cake', 'calculator', 'calendar', 'camel',
 'camera', 'camouflage', 'campfire', 'candle', 'cannon', 'canoe', 'car', 'carrot', 'castle', 'cat', 'ceiling_fan',
 'cell_phone', 'cello', 'chair', 'chandelier', 'church', 'circle', 'clarinet', 'clock', 'cloud', 'coffee_cup',
 'compass', 'computer', 'cookie', 'cooler', 'couch', 'cow', 'crab', 'crayon', 'crocodile', 'crown', 'cruise_ship',
 'cup', 'diamond', 'dishwasher', 'diving_board', 'dog', 'dolphin', 'donut', 'door', 'dragon', 'dresser',
 'drill', 'drums', 'duck', 'dumbbell', 'ear', 'elbow', 'elephant', 'envelope', 'eraser', 'eye', 'eyeglasses',
 'face', 'fan', 'feather', 'fence', 'finger', 'fire_hydrant', 'fireplace', 'firetruck', 'fish', 'flamingo',
 'flashlight', 'flip_flops', 'floor_lamp', 'flower', 'flying_saucer', 'foot', 'fork', 'frog', 'frying_pan',
 'garden', 'garden_hose', 'giraffe', 'goatee', 'golf_club', 'grapes', 'grass', 'guitar', 'hamburger',
 'hammer', 'hand', 'harp', 'hat', 'headphones', 'hedgehog', 'helicopter', 'helmet', 'hexagon', 'hockey_puck',
 'hockey_stick', 'horse', 'hospital', 'hot_air_balloon', 'hot_dog', 'hot_tub', 'hourglass', 'house', 'house_plant',
 'hurricane', 'ice_cream', 'jacket', 'jail', 'kangaroo', 'key', 'keyboard', 'knee', 'ladder', 'lantern', 'laptop',
 'leaf', 'leg', 'light_bulb', 'lighthouse', 'lightning', 'line', 'lion', 'lipstick', 'lobster', 'lollipop', 'mailbox',
 'map', 'marker', 'matches', 'megaphone', 'mermaid', 'microphone', 'microwave', 'monkey', 'moon', 'mosquito',
 'motorbike', 'mountain', 'mouse', 'moustache', 'mouth', 'mug', 'mushroom', 'nail', 'necklace', 'nose', 'ocean',
 'octagon', 'octopus', 'onion', 'oven', 'owl', 'paint_can', 'paintbrush', 'palm_tree', 'panda', 'pants',
 'paper_clip', 'parachute', 'parrot', 'passport', 'peanut', 'pear', 'peas', 'pencil', 'penguin', 'piano',
 'pickup_truck', 'picture_frame', 'pig', 'pillow', 'pineapple', 'pizza', 'pliers', 'police_car', 'pond',
 'pool', 'popsicle', 'postcard', 'potato', 'power_outlet', 'purse', 'rabbit', 'raccoon', 'radio', 'rain',
 'rainbow', 'rake', 'remote_control', 'rhinoceros', 'river', 'roller_coaster', 'rollerskates', 'sailboat',
 'sandwich', 'saw', 'saxophone', 'school_bus', 'scissors', 'scorpion', 'screwdriver', 'sea_turtle', 'see_saw',
 'shark', 'sheep', 'shoe', 'shorts', 'shovel', 'sink', 'skateboard', 'skull', 'skyscraper', 'sleeping_bag',
 'smiley_face', 'snail', 'snake', 'snorkel', 'snowflake', 'snowman', 'soccer_ball', 'sock', 'speedboat',
 'spider', 'spoon', 'spreadsheet', 'square', 'squiggle', 'squirrel', 'stairs', 'star', 'steak', 'stereo',
 'stethoscope', 'stitches', 'stop_sign', 'stove', 'strawberry', 'streetlight', 'string_bean', 'submarine',
 'suitcase', 'sun', 'swan', 'sweater', 'swing_set', 'sword', 't-shirt', 'table', 'teapot', 'teddy-bear',
 'telephone', 'television', 'tennis_racquet', 'tent', 'tiger', 'toaster', 'toe', 'toilet', 'tooth',
 'toothbrush', 'toothpaste', 'tornado', 'tractor', 'traffic_light', 'backup', 'tree', 'triangle',
 'trombone', 'truck', 'trumpet', 'umbrella', 'underwear', 'van', 'vase', 'violin', 'washing_machine',
 'watermelon', 'waterslide', 'whale', 'wheel', 'windmill', 'wine_bottle', 'wine_glass', 'wristwatch',
 'yoga', 'zebra', 'zigzag']



#---------------------------------------------------

#small dataset for debug
#CLASS_NAME = ['apple', 'bee']



def get_class_name_list():
    class_name = []
    files = glob.glob(DATA_DIR + '/train_simplified/*.csv')
    class_name = [f.split('/')[-1].replace(' ', '_').replace('.csv', '') for f in files]
    class_name.sort()
    print(class_name)
    #exit(0)


def run_simple_to_image():

    #plt.figure(figsize=(6,6))

    class_name=['apple','bee', 'cat', 'fish', 'frog', 'leaf']
    #class_name=CLASS_NAME

    for name in class_name:
        #name = 'cat'
        print('name=%s'%name)

        name = name.replace('_', ' ')
        df   = pd.read_csv(DATA_DIR + '/csv/train_simplified/%s.csv'%name)
        ## countrycode,drawing,key_id,recognized,timestamp,word

        images = []
        num = len(df)
        for n in range(num):
            key_id = df['key_id'][n]
            print('\r\t %7d/%7d   %s'%(n,num,key_id))

            point=[]
            time =[]
            strokes = eval(df['drawing'][n])
            for t,stroke in enumerate(strokes):
                x,y = stroke
                point.append(np.array((x,y),np.float32).T)
                time.append(np.full(len(x),t))

            point = np.concatenate(point).astype(np.float32)
            time  = np.concatenate(time ).astype(np.int32)

            #--------
            H,W = 32,32
            image  = np.full((H,W),0,np.uint8)

            x_max = point[:,0].max()
            x_min = point[:,0].min()
            y_max = point[:,1].max()
            y_min = point[:,1].min()
            w = x_max-x_min
            h = y_max-y_min
            #print(w,h)

            s = max(w,h)
            norm_point = (point-[x_min,y_min])/s
            norm_point = (norm_point-[w/s*0.5,h/s*0.5])*max(W,H)*0.85
            norm_point = np.floor(norm_point + [W/2,H/2]).astype(np.int32)


            #--------
            #plt.clf()
            T = time.max()+1
            for t in range(T):
                p = norm_point[time==t]
                x,y = p.T
                #plt.plot(x, y, marker='.')

                image[y,x]=255
                N = len(p)
                for i in range(N-1):
                    x0,y0 = p[i]
                    x1,y1 = p[i+1]
                    cv2.line(image,(x0,y0),(x1,y1),255,1,cv2.LINE_AA)

                # for n in range(N):
                #     x,y = p[n]
                #     #cv2.circle(image,(x,y),0,0,-1,cv2.LINE_AA)
             #--------

            images.append(image)

            #save first 100 for debug, etc
            if n < 500:
                overlay = 255-image
                image_dir = DATA_DIR+'/overlay/train_simplified/%s'%name
                os.makedirs(image_dir,exist_ok=True)
                cv2.imwrite(image_dir+'/%s.png'%key_id, overlay)

                image_show('overlay',overlay,4)
                cv2.waitKey(1)

        #-----
        print('')
        images = np.array(images)
        npy_dir = DATA_DIR+'/npy/train_simplified'
        os.makedirs(npy_dir,exist_ok=True)
        np.save(npy_dir+'/%s.npy'%name, images)
        print(images.shape)




def make_split():
    #class_name = ['apple','bee', 'cat', 'fish', 'frog', 'leaf']
    class_name = CLASS_NAME
    data_dir   = DATA_DIR

    all_dir   = data_dir + '/split/train_simplified'
    train_dir = data_dir + '/split/train_0'
    valid_dir = data_dir + '/split/valid_0'

    for name in class_name:
        name = name.replace('_', ' ')
        print(name)

        df = pd.read_csv(DATA_DIR + '/csv/train_simplified/%s.csv'%name)
        ## countrycode,drawing,key_id,recognized,timestamp,word


        key_id = df['key_id'].values.astype(np.int64)
        np.random.shuffle(key_id)

        N = len(key_id)
        N_valid = 80
        N_train = N - N_valid

        np.save( all_dir+'/%s.npy'%name, key_id)
        np.save( train_dir+'/%s.npy'%name, key_id[:N_train])
        np.save( valid_dir+'/%s.npy'%name, key_id[N_train:])


        #save as text
        #key_id = [str(i) for i in df['key_id'].values]
        #write_list_to_file(key_id, all_dir+'/%s'%name)
        #write_list_to_file(key_id[:N_train], train_dir+'/%s'%name)
        #write_list_to_file(key_id[N_train:], valid_dir+'/%s'%name)

def run_process1():

    #class_name=CLASS_NAME
    class_name=['apple','bee', 'cat', 'fish', 'frog', 'leaf']
    for name in class_name:
        #name = 'cat'
        name = name.replace('_', ' ')
        print('name=%s'%name)

        df = pd.read_csv(DATA_DIR + '/csv/train_simplified/%s.csv'%name)
        df.loc['drawing'] = df['drawing'].apply(eval)
        zz=0


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_simple_to_image()
    make_split()

    #run_process1()



