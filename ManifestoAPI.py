import requests
import xlrd
import csv


MY_API_KEY = ''
BASE_URL = 'https://manifesto-project.wzb.eu/tools/'

def api_list_core_versions(kind=None):
    request_str = BASE_URL + 'api_list_core_versions.json'
    request_str += '?api_key=' + MY_API_KEY
    if kind:
        request_str += '&kind='+kind

    response = requests.get(request_str)

    if response.headers.get('Status') != '200 OK':
        return None

    json_response = response.json()
    return json_response.get('datasets')


def api_list_metadata_versions(tag=False, details=False):
    request_str = BASE_URL + 'api_list_metadata_versions.json'
    request_str += '?api_key=' + MY_API_KEY
    if tag:
        request_str += '&tag=true'
    if details:
        request_str += '&details=true'

    response = requests.get(request_str)

    if response.headers.get('Status') != '200 OK':
        return None

    json_response = response.json()

    return json_response.get('versions')


def api_get_corpus_citation(key):
    request_str = BASE_URL + 'api_get_corpus_citation.json'
    request_str += '?key=' + key
    request_str += '&api_key=' + MY_API_KEY
    print(request_str)

    response = requests.get(request_str)

    if response.headers.get('Status') != '200 OK':
        return None

    json_response = response.json()
    return json_response.get('citation')


def api_get_core_citation(key):
    request_str = BASE_URL + 'api_get_core_citation.json'
    request_str += '?key=' + key
    request_str += '&api_key=' + MY_API_KEY

    response = requests.get(request_str)

    if response.headers.get('Status') != '200 OK':
        return None

    json_response = response.json()
    return json_response.get('citation')


def api_get_core(key, kind='dta', raw=True):
    request_str = BASE_URL + 'api_get_core.json'
    request_str += '?key=' + key
    request_str += '&kind=' + kind
    if raw:
        request_str += '&raw=true'
    else:
        request_str += '&raw=false'
    request_str += '&api_key=' + MY_API_KEY

    response = requests.get(request_str)
    if raw:
        with open(r'ManifestoDetails\cores\\'+key+'.'+kind, 'wb') as output:
            output.write(response.content)
    else:
        with open(r'ManifestoDetails\cores\\'+key+'.'+kind, 'w') as output:
            output.write(response.json().get('content'))

def api_metadata(keys, version):
    keys_str = ''
    for key in keys:
        keys_str += '&keys%5B%5D=' + key
    request_str = BASE_URL + 'api_metadata.json'
    request_str += '?api_key=' + MY_API_KEY
    request_str += keys_str
    request_str += "&version="+version

    response = requests.get(request_str)

    items = response.json().get('items')

    return items


def api_texts_and_annotations(keys, version):
    keys_str = ''
    for key in keys:
        keys_str += '&keys%5B%5D=' + key
    request_str = BASE_URL + 'api_texts_and_annotations.json'
    request_str += '?api_key=' + MY_API_KEY
    request_str += keys_str
    request_str += "&version=" + version

    response = requests.get(request_str)

    items = response.json().get('items')
    missing_items = response.json().get('missing_items')
    items_key = []
    items_kind = []
    items_items = []
    for item in items:
        items_key.append(item.get('key'))
        items_kind.append(item.get('kind'))
        items_items.append(item.get('items'))


    return items_key, items_kind, items_items, missing_items

def get_datasets_versions(write2txt=False):
    core_versions = api_list_core_versions()
    south_america_versions = api_list_core_versions(kind="south_america")

    if not write2txt:
        print('Core dataset versions:')
        for list_item in core_versions:
            print(list_item['id'], ' ==> ', list_item['name'])

        print('\nSouth America dataset versions:')
        for list_item in south_america_versions:
            print(list_item['id'], ' ==> ', list_item['name'])
    else:
        with open('../ManifestoDetails/DatasetVersions.txt', 'w') as file:
            file.write('Core dataset versions:\n')
            for list_item in core_versions:
                file.write(list_item['id'] + ' ==> ' + list_item['name'] + '\n')

            file.write('\nSouth America dataset versions:\n')
            for list_item in south_america_versions:
                file.write(list_item['id'] + ' ==> ' + list_item['name'] + '\n')



def get_metadata_versions(write2txt=False):
    metadata_versions = api_list_metadata_versions(tag=True, details=True)

    if not write2txt:
        for version in metadata_versions:
            print(version['name'])
            if 'tag' in version.keys():
                print(version['tag'])
            if 'description' in version.keys():
                print(version['description'])
            print('-' * 20)
    else:
        with open('../ManifestoDetails/MetadataVersions.txt', 'w') as file:
            for version in metadata_versions:
                file.write(version['name'] + '\n')
                if 'tag' in version.keys():
                    file.write(version['tag']+ '\n')
                if 'description' in version.keys():
                    file.write(version['description']+ '\n')
                file.write(('-' * 20) + '\n')


def get_manifesto_props(params, dataset_versions, file=None):
    result_strings = []
    for dataset_version in dataset_versions:
        xlsx_file = xlrd.open_workbook('../ManifestoDetails/cores/' + dataset_version + '.xlsx')

        sheet = xlsx_file.sheet_by_index(0)

        column_names = sheet.row_values(0)

        for row_number in range(1, sheet.nrows):
            row = sheet.row_values(row_number)
            suit = True
            for key in params.keys():
                if row[column_names.index(key)] != params[key]:
                    suit = False
                    break
            if suit:
                output_str = str(int(row[column_names.index('party')])) + '_' + str(
                    int(row[column_names.index('date')]))
                if 'countryname' in column_names:
                    output_str += " ==> " + row[column_names.index('countryname')]
                if 'partyname' in column_names:
                    output_str += " ==> " + row[column_names.index('partyname')]
                if 'pervote' in column_names:
                    output_str += " ==> " + str(float(row[column_names.index('pervote')]))
                result_strings.append(output_str)
    if not file:
        for string in result_strings:
            print(string)
    else:
        with open(file, 'w', newline='') as output_file:
            for string in result_strings:
                output_file.write(string + '\n')


def get_manifesto_metadata(params, dataset_versions, metadata_version='2018-1'):
    manifestos = []
    for dataset_version in dataset_versions:
        xlsx_file = xlrd.open_workbook(r'ManifestoDetails\cores\\' + dataset_version + '.xlsx')

        sheet = xlsx_file.sheet_by_index(0)

        column_names = sheet.row_values(0)

        for row_number in range(1, sheet.nrows):
            row = sheet.row_values(row_number)
            suit = True
            for key in params.keys():
                if key == 'date' and len(params[key]) == 4 :
                    if not row[column_names.index(key)].startswith(params[key]):
                        suit = False
                        break
                if row[column_names.index(key)] != params[key]:
                    suit = False
                    break
            if suit:
                output_str = str(int(row[column_names.index('party')])) + '_' + str(
                    int(row[column_names.index('date')]))
                manifestos.append(output_str)

    metadatas = []
    for i in range(0, len(manifestos), 50):
        mestos = manifestos[i: i + 50]
        print(len(mestos))
        metadatas += api_metadata(mestos, version=metadata_version)
    
    
    items = []
    titles = metadatas[0].keys()

    for metadata in metadatas:
        row = []
        for title in titles:
            row.append(metadata[title])
        items.append(row)    

    with open(r'ManifestoDetails\meta.csv', 'w', encoding = 'utf-8 sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(titles)
        writer.writerows(items)



def get_texts(params, metadata_file, annotated_file, nonannotated_file, version='2018-1'):
    annotated_files = []
    nonannotated_files = []
    with open(metadata_file, newline='', encoding = 'utf-8 sig') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        titles = None
        first = True
        even = False
        for row in reader:
            if first:
                titles = row
                first = False
                even = True
                continue
            if even:
                even = False
                continue
            else:
                even = True            
            manifesto_name = row[titles.index('party_id')] + '_' + row[titles.index('election_date')]
            is_satisfied = True
            for name, value in params.items():
                if name == 'election_date' and len(params[name]) == 4:
                    if not row[titles.index(name)].startswith(value):
                        is_satisfied = False
                        break
                elif row[titles.index(name)] != value:
                    is_satisfied = False
                    break

            if is_satisfied:
                if row[titles.index('annotations')] == 'True':
                    annotated_files.append(manifesto_name)
                else:
                    nonannotated_files.append(manifesto_name)
        print("Annotated: {}".format(len(annotated_files)))
        print("Not annotated: {}".format(len(nonannotated_files)))

    annotated_rows = []
    titles = ['content', 'cmp_code']
    for manifesto_name in annotated_files:
        print(manifesto_name)
        items_key, items_kind, items_items, missing_items = \
            api_texts_and_annotations([manifesto_name], version=version)
        for item in items_items[0]:
            row = [manifesto_name]
            for title in titles:
                if title == 'content' and title not in item.keys():
                    row.append(item['text'])
                else:
                    row.append(item[title])
            annotated_rows.append(row)
    titles = ['manifestoname'] + titles
    with open(annotated_file, 'w', encoding = 'utf-8 sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(titles)
        writer.writerows(annotated_rows)

    notannotated_rows = []
    titles = ['manifesto_name', 'content']
    for manifesto_name in nonannotated_files:
        print(manifesto_name)
        items_key, items_kind, items_items, missing_items = \
            api_texts_and_annotations([manifesto_name], version=version)
        notannotated_rows.append([items_key[0], items_items[0][0]['text']])

    with open(nonannotated_file, 'w', encoding = 'utf-8 sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(titles)
        writer.writerows(notannotated_rows)


# Получить версии датасета
# get_datasets_versions(write2txt=True)


# Получить версии метаданных
# get_metadata_versions(write2txt=True)


# Получить описание датасета в виде xlsx
# Нужная версия датасета берется из файла DatasetVersions.txt (Не забудь обновить этот файл!)
# api_get_core('MPDS2018b', kind='xlsx')
# api_get_core('MPDSSA2018b', kind='xlsx')


# Получить описание каждой программы, удовлетворяющей требованиям param
# params - словарь, в котором содержатся "название столбца"-"значение"
# dataset_versions - название файлов xlsx, уже скаченных
# params = {
#     "countryname": "Germany"
# }
# get_manifesto_props(params, ['MPDS2018b', 'MPDSSA2018b'], file='../NewGerman/manifestos.txt')


# Получить метаданные для каждой программы, удовлетворяющей требованиям param
# params - словарь, в котором содержатся "название столбца"-"значение"
# dataset_versions - название файлов xlsx, уже скаченных
# Последний параметр - имя csv-файла, в который будут заноситься метаданные о каждой из программ
# metadata_version - версия метаданных из файла MetadataVersions.txt (Не забудь обновить этот файл!)
# params = {
#     # "countryname": "Germany"
# }
# get_manifesto_metadata(params, ['MPDS2018b', 'MPDSSA2018b'], '../NewGerman/Metadata.csv')


# params = {
#     "language": "german",
# #     "election_date": "2017"
# }
# get_texts(params, '../NewGerman/Metadata.csv', '../NewGerman/annotated_german.csv', '../NewGerman/notannotated_german.csv')




