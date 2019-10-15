# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: dataclay/communication/grpc/generated/dataservice/dataservice.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from dataclay.communication.grpc.messages.dataservice import dataservice_messages_pb2 as dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2
from dataclay.communication.grpc.messages.common import common_messages_pb2 as dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='dataclay/communication/grpc/generated/dataservice/dataservice.proto',
  package='dataclay.communication.grpc.dataservice',
  syntax='proto3',
  serialized_options=_b('\n1dataclay.communication.grpc.generated.dataserviceB\026DataServiceGrpcService'),
  serialized_pb=_b('\nCdataclay/communication/grpc/generated/dataservice/dataservice.proto\x12\'dataclay.communication.grpc.dataservice\x1aKdataclay/communication/grpc/messages/dataservice/dataservice_messages.proto\x1a\x41\x64\x61taclay/communication/grpc/messages/common/common_messages.proto2\xf2/\n\x0b\x44\x61taService\x12\x83\x01\n\rinitBackendID\x12=.dataclay.communication.grpc.dataservice.InitBackendIDRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\xa3\x01\n\x1d\x61ssociateExecutionEnvironment\x12M.dataclay.communication.grpc.dataservice.AssociateExecutionEnvironmentRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x8b\x01\n\x11\x64\x65ployMetaClasses\x12\x41.dataclay.communication.grpc.dataservice.DeployMetaClassesRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x83\x01\n\rdeployClasses\x12=.dataclay.communication.grpc.dataservice.DeployClassesRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x7f\n\x0b\x65nrichClass\x12;.dataclay.communication.grpc.dataservice.EnrichClassRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\xa8\x01\n\x15newPersistentInstance\x12\x45.dataclay.communication.grpc.dataservice.NewPersistentInstanceRequest\x1a\x46.dataclay.communication.grpc.dataservice.NewPersistentInstanceResponse\"\x00\x12\x81\x01\n\x0cstoreObjects\x12<.dataclay.communication.grpc.dataservice.StoreObjectsRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x96\x01\n\x0fgetCopyOfObject\x12?.dataclay.communication.grpc.dataservice.GetCopyOfObjectRequest\x1a@.dataclay.communication.grpc.dataservice.GetCopyOfObjectResponse\"\x00\x12\x81\x01\n\x0cupdateObject\x12<.dataclay.communication.grpc.dataservice.UpdateObjectRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x87\x01\n\ngetObjects\x12:.dataclay.communication.grpc.dataservice.GetObjectsRequest\x1a;.dataclay.communication.grpc.dataservice.GetObjectsResponse\"\x00\x12\x7f\n\x0bnewMetaData\x12;.dataclay.communication.grpc.dataservice.NewMetaDataRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x87\x01\n\nnewVersion\x12:.dataclay.communication.grpc.dataservice.NewVersionRequest\x1a;.dataclay.communication.grpc.dataservice.NewVersionResponse\"\x00\x12\x8d\x01\n\x12\x63onsolidateVersion\x12\x42.dataclay.communication.grpc.dataservice.ConsolidateVersionRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x83\x01\n\rupsertObjects\x12=.dataclay.communication.grpc.dataservice.UpsertObjectsRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x87\x01\n\nnewReplica\x12:.dataclay.communication.grpc.dataservice.NewReplicaRequest\x1a;.dataclay.communication.grpc.dataservice.NewReplicaResponse\"\x00\x12\x8a\x01\n\x0bmoveObjects\x12;.dataclay.communication.grpc.dataservice.MoveObjectsRequest\x1a<.dataclay.communication.grpc.dataservice.MoveObjectsResponse\"\x00\x12\x90\x01\n\rremoveObjects\x12=.dataclay.communication.grpc.dataservice.RemoveObjectsRequest\x1a>.dataclay.communication.grpc.dataservice.RemoveObjectsResponse\"\x00\x12\x9d\x01\n\x18migrateObjectsToBackends\x12>.dataclay.communication.grpc.dataservice.MigrateObjectsRequest\x1a?.dataclay.communication.grpc.dataservice.MigrateObjectsResponse\"\x00\x12\xbd\x01\n\x1cgetClassIDFromObjectInMemory\x12L.dataclay.communication.grpc.dataservice.GetClassIDFromObjectInMemoryRequest\x1aM.dataclay.communication.grpc.dataservice.GetClassIDFromObjectInMemoryResponse\"\x00\x12\xa8\x01\n\x15\x65xecuteImplementation\x12\x45.dataclay.communication.grpc.dataservice.ExecuteImplementationRequest\x1a\x46.dataclay.communication.grpc.dataservice.ExecuteImplementationResponse\"\x00\x12\x85\x01\n\x0emakePersistent\x12>.dataclay.communication.grpc.dataservice.MakePersistentRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12y\n\x08\x66\x65\x64\x65rate\x12\x38.dataclay.communication.grpc.dataservice.FederateRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12}\n\nunfederate\x12:.dataclay.communication.grpc.dataservice.UnfederateRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12{\n\x06\x65xists\x12\x36.dataclay.communication.grpc.dataservice.ExistsRequest\x1a\x37.dataclay.communication.grpc.dataservice.ExistsResponse\"\x00\x12\xa2\x01\n\x13getFederatedObjects\x12\x43.dataclay.communication.grpc.dataservice.GetFederatedObjectsRequest\x1a\x44.dataclay.communication.grpc.dataservice.GetFederatedObjectsResponse\"\x00\x12\xac\x01\n\x17getReferencedObjectsIDs\x12\x46.dataclay.communication.grpc.dataservice.GetReferencedObjectIDsRequest\x1aG.dataclay.communication.grpc.dataservice.GetReferencedObjectIDsResponse\"\x00\x12\x8d\x01\n\x0c\x66ilterObject\x12<.dataclay.communication.grpc.dataservice.FilterObjectRequest\x1a=.dataclay.communication.grpc.dataservice.FilterObjectResponse\"\x00\x12{\n\tstoreToDB\x12\x39.dataclay.communication.grpc.dataservice.StoreToDBRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x84\x01\n\tgetFromDB\x12\x39.dataclay.communication.grpc.dataservice.GetFromDBRequest\x1a:.dataclay.communication.grpc.dataservice.GetFromDBResponse\"\x00\x12}\n\nupdateToDB\x12:.dataclay.communication.grpc.dataservice.UpdateToDBRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12}\n\ndeleteToDB\x12:.dataclay.communication.grpc.dataservice.DeleteToDBRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x87\x01\n\nexistsInDB\x12:.dataclay.communication.grpc.dataservice.ExistsInDBRequest\x1a;.dataclay.communication.grpc.dataservice.ExistsInDBResponse\"\x00\x12\x85\x01\n\x1c\x63leanExecutionClassDirectory\x12\x30.dataclay.communication.grpc.common.EmptyMessage\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12w\n\x0e\x63loseDbHandler\x12\x30.dataclay.communication.grpc.common.EmptyMessage\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12q\n\x08shutDown\x12\x30.dataclay.communication.grpc.common.EmptyMessage\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12}\n\x14\x64isconnectFromOthers\x12\x30.dataclay.communication.grpc.common.EmptyMessage\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x7f\n\x16registerPendingObjects\x12\x30.dataclay.communication.grpc.common.EmptyMessage\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12t\n\x0b\x63leanCaches\x12\x30.dataclay.communication.grpc.common.EmptyMessage\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x87\x01\n\x0f\x61\x63tivateTracing\x12?.dataclay.communication.grpc.dataservice.ActivateTracingRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12z\n\x11\x64\x65\x61\x63tivateTracing\x12\x30.dataclay.communication.grpc.common.EmptyMessage\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12v\n\tgetTraces\x12\x30.dataclay.communication.grpc.common.EmptyMessage\x1a\x35.dataclay.communication.grpc.common.GetTracesResponse\"\x00\x12\x89\x01\n\x10\x63loseSessionInDS\x12@.dataclay.communication.grpc.dataservice.CloseSessionInDSRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12}\n\nupdateRefs\x12:.dataclay.communication.grpc.dataservice.UpdateRefsRequest\x1a\x31.dataclay.communication.grpc.common.ExceptionInfo\"\x00\x12\x93\x01\n\x15getRetainedReferences\x12\x30.dataclay.communication.grpc.common.EmptyMessage\x1a\x46.dataclay.communication.grpc.dataservice.GetRetainedReferencesResponse\"\x00\x42K\n1dataclay.communication.grpc.generated.dataserviceB\x16\x44\x61taServiceGrpcServiceb\x06proto3')
  ,
  dependencies=[dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2.DESCRIPTOR,dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2.DESCRIPTOR,])



_sym_db.RegisterFileDescriptor(DESCRIPTOR)


DESCRIPTOR._options = None

_DATASERVICE = _descriptor.ServiceDescriptor(
  name='DataService',
  full_name='dataclay.communication.grpc.dataservice.DataService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=257,
  serialized_end=6387,
  methods=[
  _descriptor.MethodDescriptor(
    name='initBackendID',
    full_name='dataclay.communication.grpc.dataservice.DataService.initBackendID',
    index=0,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._INITBACKENDIDREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='associateExecutionEnvironment',
    full_name='dataclay.communication.grpc.dataservice.DataService.associateExecutionEnvironment',
    index=1,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._ASSOCIATEEXECUTIONENVIRONMENTREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='deployMetaClasses',
    full_name='dataclay.communication.grpc.dataservice.DataService.deployMetaClasses',
    index=2,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._DEPLOYMETACLASSESREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='deployClasses',
    full_name='dataclay.communication.grpc.dataservice.DataService.deployClasses',
    index=3,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._DEPLOYCLASSESREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='enrichClass',
    full_name='dataclay.communication.grpc.dataservice.DataService.enrichClass',
    index=4,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._ENRICHCLASSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='newPersistentInstance',
    full_name='dataclay.communication.grpc.dataservice.DataService.newPersistentInstance',
    index=5,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._NEWPERSISTENTINSTANCEREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._NEWPERSISTENTINSTANCERESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='storeObjects',
    full_name='dataclay.communication.grpc.dataservice.DataService.storeObjects',
    index=6,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._STOREOBJECTSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getCopyOfObject',
    full_name='dataclay.communication.grpc.dataservice.DataService.getCopyOfObject',
    index=7,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETCOPYOFOBJECTREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETCOPYOFOBJECTRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='updateObject',
    full_name='dataclay.communication.grpc.dataservice.DataService.updateObject',
    index=8,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._UPDATEOBJECTREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getObjects',
    full_name='dataclay.communication.grpc.dataservice.DataService.getObjects',
    index=9,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETOBJECTSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETOBJECTSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='newMetaData',
    full_name='dataclay.communication.grpc.dataservice.DataService.newMetaData',
    index=10,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._NEWMETADATAREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='newVersion',
    full_name='dataclay.communication.grpc.dataservice.DataService.newVersion',
    index=11,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._NEWVERSIONREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._NEWVERSIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='consolidateVersion',
    full_name='dataclay.communication.grpc.dataservice.DataService.consolidateVersion',
    index=12,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._CONSOLIDATEVERSIONREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='upsertObjects',
    full_name='dataclay.communication.grpc.dataservice.DataService.upsertObjects',
    index=13,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._UPSERTOBJECTSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='newReplica',
    full_name='dataclay.communication.grpc.dataservice.DataService.newReplica',
    index=14,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._NEWREPLICAREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._NEWREPLICARESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='moveObjects',
    full_name='dataclay.communication.grpc.dataservice.DataService.moveObjects',
    index=15,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._MOVEOBJECTSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._MOVEOBJECTSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='removeObjects',
    full_name='dataclay.communication.grpc.dataservice.DataService.removeObjects',
    index=16,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._REMOVEOBJECTSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._REMOVEOBJECTSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='migrateObjectsToBackends',
    full_name='dataclay.communication.grpc.dataservice.DataService.migrateObjectsToBackends',
    index=17,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._MIGRATEOBJECTSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._MIGRATEOBJECTSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getClassIDFromObjectInMemory',
    full_name='dataclay.communication.grpc.dataservice.DataService.getClassIDFromObjectInMemory',
    index=18,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETCLASSIDFROMOBJECTINMEMORYREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETCLASSIDFROMOBJECTINMEMORYRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='executeImplementation',
    full_name='dataclay.communication.grpc.dataservice.DataService.executeImplementation',
    index=19,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._EXECUTEIMPLEMENTATIONREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._EXECUTEIMPLEMENTATIONRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='makePersistent',
    full_name='dataclay.communication.grpc.dataservice.DataService.makePersistent',
    index=20,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._MAKEPERSISTENTREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='federate',
    full_name='dataclay.communication.grpc.dataservice.DataService.federate',
    index=21,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._FEDERATEREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='unfederate',
    full_name='dataclay.communication.grpc.dataservice.DataService.unfederate',
    index=22,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._UNFEDERATEREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='exists',
    full_name='dataclay.communication.grpc.dataservice.DataService.exists',
    index=23,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._EXISTSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._EXISTSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getFederatedObjects',
    full_name='dataclay.communication.grpc.dataservice.DataService.getFederatedObjects',
    index=24,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETFEDERATEDOBJECTSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETFEDERATEDOBJECTSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getReferencedObjectsIDs',
    full_name='dataclay.communication.grpc.dataservice.DataService.getReferencedObjectsIDs',
    index=25,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETREFERENCEDOBJECTIDSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETREFERENCEDOBJECTIDSRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='filterObject',
    full_name='dataclay.communication.grpc.dataservice.DataService.filterObject',
    index=26,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._FILTEROBJECTREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._FILTEROBJECTRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='storeToDB',
    full_name='dataclay.communication.grpc.dataservice.DataService.storeToDB',
    index=27,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._STORETODBREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getFromDB',
    full_name='dataclay.communication.grpc.dataservice.DataService.getFromDB',
    index=28,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETFROMDBREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETFROMDBRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='updateToDB',
    full_name='dataclay.communication.grpc.dataservice.DataService.updateToDB',
    index=29,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._UPDATETODBREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='deleteToDB',
    full_name='dataclay.communication.grpc.dataservice.DataService.deleteToDB',
    index=30,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._DELETETODBREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='existsInDB',
    full_name='dataclay.communication.grpc.dataservice.DataService.existsInDB',
    index=31,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._EXISTSINDBREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._EXISTSINDBRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='cleanExecutionClassDirectory',
    full_name='dataclay.communication.grpc.dataservice.DataService.cleanExecutionClassDirectory',
    index=32,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EMPTYMESSAGE,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='closeDbHandler',
    full_name='dataclay.communication.grpc.dataservice.DataService.closeDbHandler',
    index=33,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EMPTYMESSAGE,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='shutDown',
    full_name='dataclay.communication.grpc.dataservice.DataService.shutDown',
    index=34,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EMPTYMESSAGE,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='disconnectFromOthers',
    full_name='dataclay.communication.grpc.dataservice.DataService.disconnectFromOthers',
    index=35,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EMPTYMESSAGE,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='registerPendingObjects',
    full_name='dataclay.communication.grpc.dataservice.DataService.registerPendingObjects',
    index=36,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EMPTYMESSAGE,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='cleanCaches',
    full_name='dataclay.communication.grpc.dataservice.DataService.cleanCaches',
    index=37,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EMPTYMESSAGE,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='activateTracing',
    full_name='dataclay.communication.grpc.dataservice.DataService.activateTracing',
    index=38,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._ACTIVATETRACINGREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='deactivateTracing',
    full_name='dataclay.communication.grpc.dataservice.DataService.deactivateTracing',
    index=39,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EMPTYMESSAGE,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getTraces',
    full_name='dataclay.communication.grpc.dataservice.DataService.getTraces',
    index=40,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EMPTYMESSAGE,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._GETTRACESRESPONSE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='closeSessionInDS',
    full_name='dataclay.communication.grpc.dataservice.DataService.closeSessionInDS',
    index=41,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._CLOSESESSIONINDSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='updateRefs',
    full_name='dataclay.communication.grpc.dataservice.DataService.updateRefs',
    index=42,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._UPDATEREFSREQUEST,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EXCEPTIONINFO,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getRetainedReferences',
    full_name='dataclay.communication.grpc.dataservice.DataService.getRetainedReferences',
    index=43,
    containing_service=None,
    input_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_common_dot_common__messages__pb2._EMPTYMESSAGE,
    output_type=dataclay_dot_communication_dot_grpc_dot_messages_dot_dataservice_dot_dataservice__messages__pb2._GETRETAINEDREFERENCESRESPONSE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_DATASERVICE)

DESCRIPTOR.services_by_name['DataService'] = _DATASERVICE

# @@protoc_insertion_point(module_scope)
