class Tag(object):
    def __init__(self):
        self.static_tags = self.__load_static_tags()
        self.id2tags = self.__load_id2tags()
        self.tags2id = self.__load_tags2id()

    def array2tags(self, array):
        tags = []
        for id in array:
            tags.append(self.id2tags[id])
        return tags

    def tags2array(self, tags):
        array = []
        for tag in self.static_tags:
            if tag in tags:
                array.append(1)
            else:
                array.append(0)
        return array

    def inv_tags2array(self, array):
        tags = []
        for i, value in enumerate(array):
            if value != 0:
                tags.append(self.id2tags[i])
        return tags

    def __load_id2tags(self):
        id2tags = {}
        for i, tag in enumerate(self.static_tags):
            id2tags[i] = tag
        return id2tags

    def __load_tags2id(self):
        tags2id = {}
        for i, tag in enumerate(self.static_tags):
            tags2id[tag] = i
        return tags2id

    def __load_static_tags(self):
        static_tags_name = ['Atelectasis', 'Cardiomegaly',	'Consolidation', 'Edema',  'Enlarged Cardiomediastinum',
                            'Fracture',	'Lung Lesion',	'Lung Opacity', 	'No Finding',	'Pleural Effusion',	'Pleural Other',
                            'Pneumonia',	'Pneumothorax',  'Support Devices']

        return static_tags_name
